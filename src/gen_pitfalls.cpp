// gen_pitfalls.cpp
// Microarchitectural pitfall tests.
//
// ── Why these tests matter for cross-platform game development ───────────────
//
// Code that runs well on one ARM64 microarchitecture can be significantly
// slower on another due to differences in:
//
//   Store-to-load forwarding: when a store is immediately followed by a load
//   from the same (or overlapping) address, the CPU may forward the value from
//   the store buffer without waiting for it to reach the L1 cache. The rules
//   for when forwarding succeeds vary:
//     - Apple M1/M2/M3: forwarding works for exact size+offset match. Any
//       mismatch (e.g. writing 8 bytes, reading 4 bytes) incurs a ~10-cycle
//       penalty as the value is flushed through the cache.
//     - Snapdragon Oryon: wider forwarding — misaligned and width-mismatched
//       forwards are handled at lower cost by the memory disambiguation unit.
//     - Cortex-A78: similar to M1 in requiring exact match.
//
//   Memory barriers: DMB/DSB/ISB are always serializing to some degree, but
//   the actual cycle cost varies. On M1, a DMB ISH costs ~20 cycles; on some
//   Cortex designs it costs more. ISB (instruction barrier) is particularly
//   expensive as it flushes the pipeline.
//
//   Non-temporal stores (STNP): bypass the cache write-allocate mechanism.
//   For streaming writes at DRAM size, STNP halves the memory bus traffic by
//   eliminating the read-for-ownership (RFO) that precedes each normal store.
//   The performance gain is architecture-specific: on M1's unified memory,
//   STNP gives a significant throughput improvement; on Qualcomm with a
//   discrete LPDDR5 controller, the improvement may be larger still.
//
// ── Test structure ────────────────────────────────────────────────────────────
//
// Most tests here are latency measurements of a single serialized operation
// rather than throughput sweeps. The interesting question is "how many cycles
// does this specific hazard cost?" — a single number that directly tells you
// whether a code pattern is safe to use in a hot path.

#include "gen_pitfalls.h"
#include "gen_common.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ════════════════════════════════════════════════════════════════════════════
// Section 1: Store-to-load forwarding
// ════════════════════════════════════════════════════════════════════════════
//
// Architecture:
//   When a STORE is followed immediately by a LOAD from the same or
//   overlapping address, the CPU can "forward" the stored value from the
//   store buffer directly to the load output without waiting for the value
//   to be written to and read back from L1 cache. This is called store-to-
//   load forwarding (STL forwarding).
//
//   Forwarding rules on ARM64 vary by core:
//
//   CASE 1 — Exact match (same address, same width):
//     STR X0, [X9]  /  LDR X0, [X9]
//     Forwarding always succeeds. Latency ≈ 4–5 cycles.
//
//   CASE 2 — Width mismatch (write 64-bit, read 32-bit, same base address):
//     STR X0, [X9]  /  LDR W0, [X9]   (reads low 32 bits)
//     On M1: forwarding FAILS. The load must wait for the store to commit
//     to L1 cache, then re-load. Penalty ≈ +8–12 cycles over case 1.
//     On Oryon: forwarding may succeed or cost only a small penalty.
//
//   CASE 3 — Partial overlap (write 64-bit at X9, read 32-bit at X9+4):
//     STR X0, [X9]  /  LDR W0, [X9, #4]  (reads high 32 bits)
//     The most expensive case. The load needs bits from the stored value
//     but at a different offset. All known ARM64 cores incur a full
//     forwarding failure here.
//
// LOOP STRUCTURE for forwarding latency tests:
//   Each iteration does one STORE then one LOAD, with the load result
//   feeding the next store (via x0). This creates a serial chain where
//   each store-load pair must complete before the next can start.
//
//   The store/load slot is the 16-byte scratch area that build_loop carves
//   below the register save frame (scratch_bytes = 16, `mov x9, sp` in the
//   setup). A fixed, cache-resident, correctly owned data address — no
//   below-sp trickery needed.

// Forward declaration.
static void run_store_forwarding_tests(const BenchmarkParams& base);
static void run_barrier_tests(const BenchmarkParams& base);
static void run_nontemporal_tests(const BenchmarkParams& base, void* buf, size_t bufsz);
static void run_misaligned_tests(const BenchmarkParams& base, void* buf);
static void run_cas_tests(const BenchmarkParams& base, void* buf);
static void run_lrcpc_tests(const BenchmarkParams& base);

// ── STL forwarding loop builder ───────────────────────────────────────────────
//
// Each unrolled body:
//   store:   STR/STRW/STRH/STRB x0|w0, [x9, #0]
//   load:    LDR/LDRW/LDRH/LDRB x0|w0, [x9, #load_offset]
//   (x0 feeds the next iteration's store — genuine dependency chain)
//
// Parameters encoded at JIT compile time:
//   store_width: bytes written (8, 4, 2, 1)
//   load_width:  bytes read    (8, 4, 2, 1)
//   load_offset: byte offset for the load relative to the store address
//
// The latency measured is: store_latency + forwarding_latency (or
// forwarding_failure_penalty) per iteration.

enum class Width { B1 = 1, B2 = 2, B4 = 4, B8 = 8 };

static JitPool::TestFn build_stl_forward(uint64_t loops, uint32_t unroll,
                                         Width store_w, Width load_w,
                                         int32_t load_offset) {
    return build_loop(loops, unroll,
        [](a64::Assembler& a) {
            a.mov(x0, Imm(0x0102030405060708ULL));  // non-trivial initial value
            a.mov(x9, sp);                          // x9 = scratch slot address
        },
        [store_w, load_w, load_offset](a64::Assembler& a, uint32_t) {
            // Store x0/w0 at [x9].
            switch (store_w) {
                case Width::B8: a.str (x0,  ptr(x9)); break;
                case Width::B4: a.str (w0,  ptr(x9)); break;
                case Width::B2: a.strh(w0,  ptr(x9)); break;
                case Width::B1: a.strb(w0,  ptr(x9)); break;
            }
            // Load from [x9 + load_offset] into x0/w0.
            // The load width determines how many bytes of the stored value are read.
            if (load_offset == 0) {
                switch (load_w) {
                    case Width::B8: a.ldr  (x0, ptr(x9)); break;
                    case Width::B4: a.ldr  (w0, ptr(x9)); break;
                    case Width::B2: a.ldrh (w0, ptr(x9)); break;
                    case Width::B1: a.ldrb (w0, ptr(x9)); break;
                }
            } else {
                switch (load_w) {
                    case Width::B8: a.ldr  (x0, ptr(x9, load_offset)); break;
                    case Width::B4: a.ldr  (w0, ptr(x9, load_offset)); break;
                    case Width::B2: a.ldrh (w0, ptr(x9, load_offset)); break;
                    case Width::B1: a.ldrb (w0, ptr(x9, load_offset)); break;
                }
            }
        },
        /*scratch_bytes=*/16);
}

static void run_store_forwarding_tests(const BenchmarkParams& base) {
    section("Store-to-load forwarding");
    printf("  clk/insn = latency of one STORE+LOAD pair (unroll=8).\n"
           "  M1 penalty for mismatch: ~+8–12 clk vs matched case.\n\n");

    // Loop counts: forwarding latency ≈ 4–16 cycles → target ~100ms.
    // At 15 cyc × (1/3.2 GHz) × 8 unroll = ~37.5ns/iter → 100ms/37.5ns ≈ 2.7M.
    const uint64_t loops  = scale_loops(3'000'000);
    const uint32_t unroll = 8;
    char name[80];

    struct FwdCase {
        const char* label;
        Width       store_w;
        Width       load_w;
        int32_t     load_offset;
    };

    const FwdCase cases[] = {
        // ── Baseline: exact 64-bit match ─────────────────────────────────
        // Should always forward successfully on all ARM64 cores.
        { "STR x64 → LDR x64   (matched,   offset 0)",
          Width::B8, Width::B8, 0 },

        // ── Exact 32-bit match ────────────────────────────────────────────
        { "STR w32 → LDR w32   (matched,   offset 0)",
          Width::B4, Width::B4, 0 },

        // ── Width mismatch: write 64, read 32 (low half) ─────────────────
        // Reads the low 32 bits of the 64-bit stored value.
        // M1: forwarding failure expected (~+8–12 cycles vs matched).
        // Oryon: may succeed at lower cost.
        { "STR x64 → LDR w32   (mismatch,  offset 0)",
          Width::B8, Width::B4, 0 },

        // ── Width mismatch: write 64, read 32 (high half) ────────────────
        // Offset +4: reads the upper 32 bits of a 64-bit store.
        // PARTIAL OVERLAP — almost certainly fails on all cores.
        { "STR x64 → LDR w32   (overlap,   offset +4)",
          Width::B8, Width::B4, 4 },

        // ── Width mismatch: write 32, read 64 ────────────────────────────
        // Tries to read 8 bytes when only 4 were stored. The high 4 bytes
        // have undefined content from whatever was in memory before.
        // ARM64 allows this (no fault), but forwarding will fail.
        { "STR w32 → LDR x64   (narrow→wide, offset 0)",
          Width::B4, Width::B8, 0 },

        // ── Width mismatch: write 64, read 8 ─────────────────────────────
        { "STR x64 → LDRB w8   (mismatch,  offset 0)",
          Width::B8, Width::B1, 0 },

        // ── Exact 8-bit match ─────────────────────────────────────────────
        { "STRB w8  → LDRB w8  (matched,   offset 0)",
          Width::B1, Width::B1, 0 },
    };

    for (const auto& c : cases) {
        auto fn = build_stl_forward(loops, unroll, c.store_w, c.load_w,
                                    c.load_offset);
        snprintf(name, sizeof(name), "%-46s", c.label);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: Memory ordering barriers
// ════════════════════════════════════════════════════════════════════════════
//
// ARM64 memory model is weakly ordered: loads and stores can reorder with
// respect to each other. Explicit barriers prevent this:
//
//   DMB ISH (Data Memory Barrier, Inner Shareable):
//     Ensures all memory accesses before the barrier complete before any
//     memory accesses after the barrier. The most common barrier in
//     lock-free code. "Inner shareable" covers all cores in the same
//     cluster — sufficient for SMP within a single chip.
//
//   DSB ISH (Data Synchronization Barrier):
//     Stronger than DMB: not only orders memory accesses, but also
//     ensures all cache maintenance and TLB operations complete.
//     Used before context switches and page table walks. More expensive.
//
//   ISB (Instruction Synchronization Barrier):
//     Flushes the instruction pipeline. Ensures instructions fetched
//     after the ISB reflect any changes to system registers, cache
//     state, or instruction memory made before it. Very expensive.
//     Required after self-modifying code (relevant to JIT engines).
//
//   LDAR (Load-Acquire Register):
//     A load with acquire semantics built into the instruction — no
//     separate barrier needed. LDAR prevents loads/stores after the LDAR
//     from being reordered before it. Preferred over LDR+DMB because it
//     can be implemented with a single instruction and often at lower
//     latency than a separate DMB.
//
// METHODOLOGY:
//   We measure the cost of each barrier in isolation by putting N copies
//   per loop iteration. For latency, we also create a dependency chain:
//   the result of the previous iteration's final instruction feeds the
//   address used in the next iteration, preventing the CPU from running
//   multiple iterations concurrently.

static void run_barrier_tests(const BenchmarkParams& base) {
    section("Memory ordering barriers");
    printf("  clk/insn = cycles per barrier instruction (unroll=8).\n\n");

    const uint64_t loops  = scale_loops(3'000'000);
    const uint32_t unroll = 8;
    char name[80];

    // ── Shuffled L1 pointer ring ───────────────────────────────────────────
    // A 64-node ring (512 bytes, fits in L1) linked in shuffled order so that
    // every load returns a *different* address. Placed on the stack so it
    // stays alive for the entire run_barrier_tests call.
    //
    // WHY NOT a self-referential chain ([x9] = x9)?
    //   On Apple M5 (and possibly earlier), a chain where every load always
    //   returns the same value (itself) is defeated by load value prediction:
    //   the CPU learns the constant result and "executes" the loads in 0 cycles.
    //   A shuffled ring visits 64 distinct addresses, defeating value predictors.
    //   LDAR/LDAPR/LDAR are unaffected — their ordering semantics prevent
    //   speculative value use regardless of chain shape.
    constexpr uint32_t kRingN = 64;
    alignas(64) uintptr_t ring_buf[kRingN];
    void* const ring_start = build_pointer_ring(ring_buf, sizeof(ring_buf),
                                                sizeof(uintptr_t));
    if (!ring_start) return;
    const uint64_t ring_head = reinterpret_cast<uint64_t>(ring_start);

    // ── LDR baseline (plain load, no ordering) ────────────────────────────
    // Pointer-chase through the shuffled L1 ring. This is the true L1 load
    // latency reference; a barrier adds overhead on top of this.
    {
        auto fn = build_loop(loops, unroll,
            [ring_head](a64::Assembler& a)  { a.mov(x0, Imm(ring_head)); },
            [](a64::Assembler& a, uint32_t) { a.ldr(x0, ptr(x0)); });
        snprintf(name, sizeof(name), "LDR x64 (L1 chain, baseline)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── LDAR: load-acquire ────────────────────────────────────────────────
    // Same ring, LDAR instead of LDR. LDAR prevents reordering of later
    // accesses before this load and inhibits load value speculation.
    {
        auto fn = build_loop(loops, unroll,
            [ring_head](a64::Assembler& a)  { a.mov(x0, Imm(ring_head)); },
            [](a64::Assembler& a, uint32_t) { a.ldar(x0, ptr(x0)); });
        snprintf(name, sizeof(name), "LDAR x64 (load-acquire)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── DMB ISH ───────────────────────────────────────────────────────────
    // A stream of DMB ISH instructions with no surrounding loads/stores.
    // This measures the raw barrier serialization cost.
    // In practice a DMB always occurs between memory accesses, so this is
    // a lower bound on the cost it adds to lock/unlock operations.
    {
        auto fn = build_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { a.dmb(Imm(Predicate::DB::kISH)); });
        snprintf(name, sizeof(name), "DMB ISH");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── DSB ISH ───────────────────────────────────────────────────────────
    {
        auto fn = build_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { a.dsb(Imm(Predicate::DB::kISH)); });
        snprintf(name, sizeof(name), "DSB ISH");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── ISB (Instruction Synchronization Barrier) ─────────────────────────
    // Flushes and refills the instruction pipeline. The most expensive
    // barrier — used only when instruction cache coherency is required
    // (e.g. after writing JIT code into executable memory).
    // We expect this to be substantially more expensive than DMB/DSB.
    {
        auto fn = build_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { a.isb(Imm(0xF)); });  // 0xF = SY option
        snprintf(name, sizeof(name), "ISB SY");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── LDR + DMB ISH (acquire pattern) vs LDAR ───────────────────────────
    // The traditional way to implement a load-acquire is LDR followed by DMB.
    // LDAR is the preferred single-instruction equivalent.
    // If LDR+DMB total cost ≈ LDAR cost, the hardware is folding them.
    // If LDR+DMB is more expensive, LDAR is genuinely faster.
    // Uses the same shuffled L1 ring as the LDR baseline.
    {
        const uint32_t pair_unroll = 4;  // 4 pairs = 8 instructions
        auto fn = build_loop(loops, pair_unroll,
            [ring_head](a64::Assembler& a) { a.mov(x0, Imm(ring_head)); },
            [](a64::Assembler& a, uint32_t) {
                a.ldr(x0, ptr(x0));
                a.dmb(Imm(Predicate::DB::kISH));
            });
        snprintf(name, sizeof(name), "LDR + DMB ISH (manual acquire, 4 pairs)");
        run_one(name, fn, params_for(base, loops, pair_unroll * 2));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: Non-temporal stores (STNP)
// ════════════════════════════════════════════════════════════════════════════
//
// STNP (Store Non-Temporal Pair) hints to the CPU that the stored data will
// not be accessed again soon. The CPU MAY bypass the cache and write directly
// to memory. This eliminates the read-for-ownership (RFO) traffic that a
// normal store generates: normally, before writing a cache line, the CPU must
// first read the existing line into cache (the "read" part of "read-modify-
// write"). STNP says "skip the read, just write."
//
// For streaming writes to large buffers (particle systems, vertex uploads,
// render target clears, audio mixing output), STNP can nearly double
// effective write bandwidth:
//   Normal STP to DRAM: ≈ 28 GB/s (from gen_memory results)
//   Expected STNP:      ≈ 40–55 GB/s (no RFO = half the DRAM traffic)
//
// IMPORTANT CAVEAT: STNP is a HINT, not a guarantee. The CPU is free to
// treat it as a normal store. On M1, Apple's implementation does honour the
// hint for large buffers; on some Cortex designs STNP is effectively a no-op.
// Comparing STNP bandwidth to STP bandwidth directly tells you whether the
// hint is implemented on the current CPU.
//
// We reuse the gen_memory bandwidth loop structure: an outer loop over passes
// and an inner loop over kBwStep-byte blocks, using 4 STP/STNP per cache line.

static constexpr size_t kNTCacheLine = 64;
static constexpr size_t kNTStep      = 512;   // 8 cache lines per step
static constexpr uint32_t kNTLines   = static_cast<uint32_t>(kNTStep / kNTCacheLine);

static JitPool::TestFn build_stnp_bw(uintptr_t buf_base, size_t buf_size,
                                      uint64_t num_passes, bool non_temporal) {
    const uint64_t inner_iters = buf_size / kNTStep;

    // Outer loop = one pass over the buffer (build_loop, unroll 1); the body
    // emits the inner loop that walks the buffer kNTStep bytes at a time.
    //   x20 = buffer base, x10 = store value, x0 = cursor, x21 = inner counter
    return build_loop(num_passes, 1,
        [buf_base](a64::Assembler& a) {
            a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));
            a.mov(x10, x20);  // store value (non-trivial = buf_base)
        },
        [inner_iters, non_temporal](a64::Assembler& a, uint32_t) {
            a.mov(x0,  x20);
            a.mov(x21, Imm(inner_iters));

            Label inner = a.new_label();
            a.bind(inner);

            for (uint32_t line = 0; line < kNTLines; ++line) {
                const int32_t off = static_cast<int32_t>(line * kNTCacheLine);
                if (non_temporal) {
                    a.stnp(x10, x10, ptr(x0, off + 0));
                    a.stnp(x10, x10, ptr(x0, off + 16));
                    a.stnp(x10, x10, ptr(x0, off + 32));
                    a.stnp(x10, x10, ptr(x0, off + 48));
                } else {
                    a.stp(x10, x10, ptr(x0, off + 0));
                    a.stp(x10, x10, ptr(x0, off + 16));
                    a.stp(x10, x10, ptr(x0, off + 32));
                    a.stp(x10, x10, ptr(x0, off + 48));
                }
            }

            a.add(x0, x0, Imm(static_cast<uint64_t>(kNTStep)));
            a.sub(x21, x21, Imm(1));
            a.cbnz(x21, inner);
        });
}

static void run_nontemporal_tests(const BenchmarkParams& base,
                                  void* buf, size_t bufsz) {
    section("Non-temporal stores (STNP vs STP)");
    printf("  Bandwidth in GB/s. STNP hint bypasses write-allocate RFO.\n"
           "  If STNP ≈ STP: hint not honoured (treated as normal store).\n"
           "  If STNP > STP: hint works; less DRAM traffic from RFO bypass.\n\n");

    // Test at three buffer sizes: L1 (in-cache), L2 (in-cache), DRAM.
    struct NTSize { size_t bytes; const char* label; };
    const NTSize sizes[] = {
        { 64ULL  * 1024,    "  64KB (L1)" },
        { 4ULL   * 1024*1024, "   4MB (L2)" },
        { 64ULL  * 1024*1024, "  64MB (DRAM)" },
    };

    for (const auto& sz : sizes) {
        if (sz.bytes > bufsz) continue;
        const uint64_t lines     = sz.bytes / kNTCacheLine;
        const uint64_t passes    = 50'000'000 / lines;
        const uint64_t p_clamped = scale_loops(
            (passes < 4) ? 4 : (passes > 2'000'000 ? 2'000'000 : passes));

        const BenchmarkParams p = params_for(base, p_clamped,
                                             static_cast<uint32_t>(lines),
                                             static_cast<uint32_t>(kNTCacheLine));

        char name[64];

        // STP reference
        auto fn_stp = build_stnp_bw(
            reinterpret_cast<uintptr_t>(buf), sz.bytes, p_clamped, false);
        snprintf(name, sizeof(name), "STP  (normal store)    %s", sz.label);
        run_one(name, fn_stp, p);

        // STNP non-temporal
        auto fn_nt = build_stnp_bw(
            reinterpret_cast<uintptr_t>(buf), sz.bytes, p_clamped, true);
        snprintf(name, sizeof(name), "STNP (non-temporal)    %s", sz.label);
        run_one(name, fn_nt, p);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 4: Misaligned load penalty
// ════════════════════════════════════════════════════════════════════════════
//
// ARM64 supports hardware-assisted unaligned loads and stores: a load that
// straddles a cache line or page boundary will not fault. However, the
// hardware cost differs by alignment:
//
//   Within a cache line (offset < 64): usually free or 1 cycle penalty.
//   Crossing a cache line (e.g. reading 8 bytes at offset 60 within a line):
//     costs an extra cache-line fetch. Penalty ≈ 1–3 cycles.
//   Crossing a page boundary (4096-byte boundary):
//     may require two TLB lookups. Penalty ≈ 10–30 cycles on some cores.
//
// We measure load latency (pointer-chasing) from addresses with different
// alignment offsets within a pre-allocated buffer. The pointer chain is set
// up so every pointer in the chain uses the same misalignment offset,
// making the measurement representative.

static void run_misaligned_tests(const BenchmarkParams& base, void* buf) {
    section("Misaligned load latency");
    printf("  Pointer-chase through a buffer; each pointer is misaligned\n"
           "  by the given byte offset from 8-byte alignment.\n\n");

    const uint64_t loops   = scale_loops(10'000'000);
    char name[80];

    // Offsets to test. 0 = naturally aligned. Others probe crossing points.
    const int32_t offsets[] = { 0, 1, 4, 7, 56, 60, 63 };

    for (int32_t off : offsets) {
        // Shuffled chain of nodes at (naturally_aligned_addr + off), one node
        // every 256 bytes over a 2MB window (fits in L2). build_pointer_ring
        // writes the links with memcpy — at a nonzero offset the link slot is
        // deliberately unaligned, and a direct pointer store there is UB.
        const size_t stride   = 256;
        const size_t buf_size = 2ULL * 1024 * 1024;  // 2MB — fits in L2

        void* const head = build_pointer_ring(buf, buf_size, stride,
                                              static_cast<size_t>(off));
        if (!head) continue;
        const uint64_t head_addr = reinterpret_cast<uint64_t>(head);

        auto fn = build_loop(loops, 1,
            [head_addr](a64::Assembler& a)  { a.mov(x0, Imm(head_addr)); },
            [](a64::Assembler& a, uint32_t) { a.ldr(x0, ptr(x0)); });

        const char* boundary = (off == 0)    ? "(aligned)"        :
                               (off < 8)     ? "(within 8B word)" :
                               (off <= 55)   ? "(within cache line)" :
                               (off <= 63)   ? "(crosses cache line)" : "";
        snprintf(name, sizeof(name), "LDR misalign +%2d bytes %s", off, boundary);
        run_one(name, fn, params_for(base, loops, 1));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 5: CAS (Compare-and-Swap) latency
// ════════════════════════════════════════════════════════════════════════════
//
// CAS is the primitive operation underlying most lock-free data structures
// and spinlocks. Its single-threaded latency establishes the minimum cost
// of a spin-lock acquire+release cycle when there is no contention.
//
// ARM64 CAS (from ARMv8.1 Large System Extensions):
//   CAS Xs, Xt, [Xn]  — atomically: if [Xn]==Xs, then [Xn]=Xt
//
// Latency includes: load from cache, compare, conditional store, and
// whatever ordering semantics the variant implies.
//
// We test three CAS variants:
//   CAS   (relaxed — no ordering)
//   CASA  (acquire — prevents later loads/stores being reordered before it)
//   CASAL (acquire+release — full sequential consistency)
//
// For a spinlock, CASAL is the correct choice. Its latency directly determines
// the maximum lock/unlock frequency when the lock is uncontended.

static void run_cas_tests(const BenchmarkParams& base, void* buf) {
    section("CAS (Compare-and-Swap) latency");
    printf("  Single-threaded CAS on an L1-resident cache line.\n"
           "  clk/insn = total CAS round-trip latency (load+compare+store).\n\n");

    const uint64_t loops  = scale_loops(5'000'000);
    const uint32_t unroll = 4;
    char name[80];

    // The CAS target is a single 8-byte word in the buffer.
    uintptr_t cas_addr = reinterpret_cast<uintptr_t>(buf);
    // Align to cache line to ensure it's not straddling two lines.
    cas_addr = (cas_addr + 63) & ~static_cast<uintptr_t>(63);

    // Pre-initialise the target word to 0.
    *reinterpret_cast<uint64_t*>(cas_addr) = 0;

    // ── CAS relaxed ───────────────────────────────────────────────────────
    // CAS: if [x9] == x1 (expected), write x2 (new) to [x9].
    // We keep expected=0 and new=0, so CAS always succeeds and leaves
    // [x9]=0. The memory ordering of each CAS's store must complete before
    // the next CAS can confirm [x9]==0, serializing all iterations.
    {
        auto fn = build_loop(loops, unroll,
            [cas_addr](a64::Assembler& a) {
                a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
                a.mov(x2, Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.mov(x1, Imm(0));
                a.cas(x1, x2, ptr(x9));
            });
        snprintf(name, sizeof(name), "CAS   x64 relaxed (always succeeds)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CASAL (acquire+release) ───────────────────────────────────────────
    // Full sequential-consistency CAS. This is what a correct spinlock
    // acquire needs. Its latency is the minimum uncontended lock cycle time.
    {
        auto fn = build_loop(loops, unroll,
            [cas_addr](a64::Assembler& a) {
                a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
                a.mov(x2, Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.mov(x1, Imm(0));
                a.casal(x1, x2, ptr(x9));
            });
        snprintf(name, sizeof(name), "CASAL x64 acq+rel (spinlock acquire)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── LDAXR + STLXR (LL/SC, acquire+release) ───────────────────────────
    // The traditional load-linked / store-conditional spinlock pattern.
    // Comparing with CASAL reveals whether the hardware fuses CASAL into
    // a single micro-op or expands it to an LL/SC internally.
    //
    // NOTE: No retry loop here. On macOS the OS scheduler may preempt a
    // thread mid-exclusive-monitor window and clear the reservation, causing
    // STLXR to fail. A retry loop would then livelock indefinitely.
    // Instead we proceed regardless of the STLXR status bit (w2). In the
    // rare case of a failed SC, that iteration measures slightly higher
    // latency — this appears as noise in our CoV rather than an infinite
    // loop. The benchmark is for *latency*, not for correctness of the
    // store; the LDAXR ordering cost is what we're measuring.
    {
        auto fn = build_loop(loops, unroll,
            [cas_addr](a64::Assembler& a) {
                a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
                a.mov(x0, Imm(0));   // new value = 0
            },
            [](a64::Assembler& a, uint32_t) {
                a.ldaxr(x1, ptr(x9));      // load-acquire-exclusive: x1 = [x9]
                a.stlxr(w2, x0, ptr(x9)); // store-release-exclusive: [x9] = 0
                // w2 = 0 on success, 1 on failure. We don't check or retry.
                // The LDAXR→STLXR window contains zero other instructions,
                // minimising the chance of preemption breaking the reservation.
            });
        snprintf(name, sizeof(name), "LDAXR+STLXR (LL/SC, no-retry)");
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 6: LRCPC load-acquire variants (FEAT_LRCPC / FEAT_LRCPC2)
// ════════════════════════════════════════════════════════════════════════════
//
// FEAT_LRCPC (ARMv8.3) adds LDAPR: Load-Acquire RCpc Register.
// "RCpc" = Release Consistency Processor Consistent — a weaker acquire than
// LDAR: it is only a one-way barrier (prevents later loads from being observed
// before the LDAPR), but does NOT prevent stores from completing afterward.
// This matches the x86/TSO memory model's load semantics exactly.
//
// FEAT_LRCPC2 (ARMv8.4) adds LDAPUR and STLUR: unscaled-offset variants of
// LDAPR and STLR, enabling use from arbitrary offsets without a prior ADD.
//
// WHY THIS MATTERS — the FEX 50% speedup:
//   FEX-Emu (x86-on-ARM64 translator) used LDAR for x86 load emulation.
//   Switching to LDAPUR (correct weaker semantic, no full-barrier overhead)
//   gave a ~50% speedup overnight on some workloads. The key is that LDAPUR
//   does not need to drain the store buffer before completing, unlike LDAR.
//
// HARDWARE BUG DETECTOR:
//   Some ARM cores have incorrect LRCPC implementations: they treat
//   LDAPR/LDAPUR as full LDAR internally, eliminating the benefit.
//   The tests here expose this:
//
//   LDAPR latency ≈ LDR latency  → correct, one-way barrier is nearly free
//   LDAPR latency ≈ LDAR latency → buggy, treated as full acquire barrier
//
//   Additionally, if LDAPR and LDAPUR give *different* latencies, the two
//   instruction forms are going through different pipelines — a separate
//   implementation quality signal.
//
// STORE-TO-LOAD FORWARDING with LRCPC:
//   Does adding release/acquire ordering to a store+load pair affect whether
//   the value forwards through the store buffer? Four combinations:
//     STR  → LDAPR  (relaxed store, LRCPC load)
//     STLR → LDAPR  (release store, LRCPC load)
//     STR  → LDAPUR (relaxed store, LRCPC2 load)
//     STLUR→ LDAPUR (release store, LRCPC2 load)
//   Compare to STLR → LDAR from the barrier section (full ordered pair).
//   If forwarding is preserved: ordering semantics don't inhibit the store
//   buffer bypass. If latency rises: the barrier drained the store buffer.

static void run_lrcpc_tests(const BenchmarkParams& base) {
    const bool lrcpc  = cpu_has(CpuFeature::LRCPC);
    const bool lrcpc2 = cpu_has(CpuFeature::LRCPC2);

    if (!lrcpc && !lrcpc2) {
        section("LRCPC load-acquire (FEAT_LRCPC / FEAT_LRCPC2)");
        skip_feature(CpuFeature::LRCPC, "LDAPR/LDAPUR tests");
        return;
    }

    section("LRCPC load-acquire (FEAT_LRCPC / FEAT_LRCPC2)");
    printf("  LDAPR (FEAT_LRCPC): one-way acquire barrier (weaker than LDAR).\n"
           "  On CPUs where LDAR > LDR: correct LDAPR ≈ LDR, buggy ≈ LDAR.\n"
           "  On Apple M-series: LDAR = LDR (no store to drain in pointer chain),\n"
           "    so LDAPR = LDAR = LDR is CORRECT — check forwarding tests below.\n"
           "  LDAPUR/STLUR (FEAT_LRCPC2): unscaled-offset variants.\n\n");

    const uint64_t loops  = scale_loops(3'000'000);
    const uint32_t unroll = 8;
    char name[80];

    // ── LDAPR pointer chain (FEAT_LRCPC) ─────────────────────────────────
    // Pointer chase using LDAPR instead of LDR/LDAR.
    // Compare to: "LDR x64 (L1 chain, baseline)" and "LDAR x64 (load-acquire)"
    // from the barrier section above.
    //
    // A self-referential slot ([x9] = x9) is safe here: LDAPR's ordering
    // semantics require the load to actually complete before the result is
    // consumable, so the load value predictor cannot short-circuit it.
    if (lrcpc) {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x9, sp);
                a.str(x9, ptr(x9));
                a.ldapr(x0, ptr(x9));  // prime
            },
            [](a64::Assembler& a, uint32_t) { a.ldapr(x0, ptr(x0)); },
            /*scratch_bytes=*/16);
        snprintf(name, sizeof(name), "LDAPR  x64 (FEAT_LRCPC,  L1 chain)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── LDAPUR pointer chain (FEAT_LRCPC2) ───────────────────────────────
    // LDAPUR Xt, [Xn, #0]: same ordering semantics as LDAPR, unscaled offset.
    // On correct hardware: identical latency to LDAPR.
    // If LDAPUR ≠ LDAPR: the two forms are on different pipelines.
    if (lrcpc2) {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x9, sp);
                a.str(x9, ptr(x9));
                a.ldapur(x0, ptr(x9));  // prime
            },
            [](a64::Assembler& a, uint32_t) { a.ldapur(x0, ptr(x0)); },
            /*scratch_bytes=*/16);
        snprintf(name, sizeof(name), "LDAPUR x64 (FEAT_LRCPC2, L1 chain)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── Store-to-load forwarding with LRCPC instructions ─────────────────
    // Same scratch-slot methodology as run_store_forwarding_tests.
    // x9 = pointer to the 16-byte scratch slot; x0 is the value being forwarded.

    printf("\n  LRCPC store-to-load forwarding (compare to STR→LDR baseline above):\n\n");

    struct LrcpcFwdCase {
        const char* label;
        bool        need_lrcpc2_store;   // STLUR vs STLR
        bool        need_lrcpc2_load;    // LDAPUR vs LDAPR
    };

    // Four combinations: {relaxed,release} store × {LDAPR,LDAPUR} load.
    const LrcpcFwdCase fwd_cases[] = {
        { "STR   → LDAPR  (relaxed→rcpc-acq)", false, false },
        { "STLR  → LDAPR  (release→rcpc-acq)", false, false },  // uses STLR
        { "STR   → LDAPUR (relaxed→rcpc2-acq)", false, true },
        { "STLUR → LDAPUR (release→rcpc2-acq)", true,  true },
    };

    for (uint32_t ci = 0; ci < 4; ++ci) {
        const auto& c = fwd_cases[ci];
        const bool use_stlr  = (ci == 1);           // STLR (no offset, FEAT_V8)
        const bool use_stlur = c.need_lrcpc2_store;  // STLUR (FEAT_LRCPC2)
        const bool use_ldapur = c.need_lrcpc2_load;  // LDAPUR vs LDAPR

        if ((use_stlur || use_ldapur) && !lrcpc2) continue;
        if (!use_stlur && !use_ldapur && !lrcpc)  continue;

        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0x0102030405060708ULL));
                a.mov(x9, sp);
            },
            [use_stlr, use_stlur, use_ldapur](a64::Assembler& a, uint32_t) {
                // Store.
                if (use_stlur)     a.stlur(x0, ptr(x9));
                else if (use_stlr) a.stlr(x0, ptr(x9));
                else               a.str (x0, ptr(x9));
                // Load.
                if (use_ldapur) a.ldapur(x0, ptr(x9));
                else            a.ldapr(x0, ptr(x9));
            },
            /*scratch_bytes=*/16);
        snprintf(name, sizeof(name), "%-46s", c.label);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 7: BFI destination dependency (Mihocka stress)
// ════════════════════════════════════════════════════════════════════════════
//
// BFI inherently reads its destination register to preserve the bits it does
// not overwrite. A naive latency chain — `bfi x0, x1, #pos, #w` repeated with
// the same destination — therefore carries a true dep through Xd. But a clever
// micro-architecture could in principle break the dep when it can prove the
// non-inserted bits don't matter (e.g., when the next instruction overwrites
// the same range, or when the inserted range is full-width).
//
// Darek Mihocka (Prism, Microsoft) suggested a stress test that defeats any
// such heuristic by *rotating* the chain across multiple registers with
// *overlapping* bit positions, so the µarch can never prove a previous
// destination is dead. This file implements three variants:
//
//   Variant A — independent BFI (throughput baseline):
//     bfi x0, x10, #0, #8         ; x10 is a stable constant
//     bfi x0, x10, #0, #8         ; same dest, no inter-instruction chain
//     ...
//     Reports throughput, not latency. Apple M / Cortex-X expected ≈ 0.25–
//     0.5 clk/insn (multiple BFI-capable ALUs).
//
//   Variant B — overlapping rotated chain (true latency):
//     bfi x1, x0, #1, #2          ; x1[2:1] ← x0[1:0]
//     bfi x2, x1, #1, #2          ; x2[2:1] ← x1[1:0]   (overlaps x1 update)
//     bfi x0, x2, #1, #2          ; x0[2:1] ← x2[1:0]   (closes the loop)
//     ...
//     The 2-bit insert at position 1 overlaps the source's 2-bit read at
//     position 0 (bit 1 is both written by the previous BFI and read as
//     part of the next BFI's source). No dep-breaking heuristic can apply.
//     Reports true BFI latency. Apple M / Cortex-X / Snapdragon X+ ≈ 1 clk;
//     pre-X1 Cortex-A and pre-Oryon Snapdragon ≈ 2 clk.
//
//   Variant C — full-width destructive write:
//     bfi x0, x10, #0, #64
//     This degenerates to a plain MOV (all destination bits replaced).
//     Whether the µarch recognises this and breaks the Xd dep is the
//     interesting question. Most don't bother (BFI with width=64 is rare),
//     so expect ~1 clk/insn just like Variant B; if a chip *does* dep-break,
//     it'll show ~0 clk added beyond loop overhead.
//
// All instructions are baseline ARMv8.0; no runtime feature detection needed.

static void run_bfi_dependency_tests(const BenchmarkParams& base) {
    section("BFI destination dependency (Mihocka stress)");
    printf("  Variant A: throughput baseline (no Xd chain)\n");
    printf("  Variant B: overlapping bitfield rotation across x0/x1/x2 (true latency)\n");
    printf("  Variant C: full-width BFI (degenerates to MOV — does µarch dep-break?)\n\n");

    const uint64_t loops  = scale_loops(5'000'000);
    const uint32_t unroll = 24;            // multiple of 3 for variant B rotation
    char name[80];

    // ── Variant A: independent BFI — throughput-bound ─────────────────────
    // All instructions in the unrolled body use the same Xd (x0) and same
    // source (x10). x0 is overwritten in the same slice each iteration, so
    // there is no inter-instruction Xd dependency — only the implicit RMW
    // dependency on x0 that the µarch *might* be able to break.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0xDEADBEEFCAFEBABEULL));
                a.mov(x10, Imm(0x1234567890ABCDEFULL));
            },
            [](a64::Assembler& a, uint32_t) {
                a.bfi(x0, x10, Imm(0), Imm(8));
            });
        snprintf(name, sizeof(name), "BFI variant A (independent, x0 RMW)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── Variant B: overlapping rotated chain — true latency ───────────────
    // Rotate the chain across three registers with bit positions that
    // genuinely overlap. The 2-bit insert at #1 means bit 1 is both written
    // by the previous BFI's output and read as part of the next BFI's
    // source — a real read-after-write dependency that no µarch heuristic
    // can break.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0xAAAAAAAAAAAAAAAAULL));
                a.mov(x1, Imm(0x5555555555555555ULL));
                a.mov(x2, Imm(0xF0F0F0F0F0F0F0F0ULL));
            },
            [](a64::Assembler& a, uint32_t u) {
                // u%3==0: bfi x1, x0
                // u%3==1: bfi x2, x1
                // u%3==2: bfi x0, x2
                static const a64::Gp dst[3] = { x1, x2, x0 };
                static const a64::Gp src[3] = { x0, x1, x2 };
                a.bfi(dst[u % 3], src[u % 3], Imm(1), Imm(2));
            });
        snprintf(name, sizeof(name), "BFI variant B (overlapping rotation, true lat)");
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── Variant C: full-width BFI (lsb=0, width=64) — equivalent to MOV ───
    // Every bit of x0 is replaced. A µarch that recognises this could break
    // the Xd dep entirely. Most don't bother. AsmJit may reject width=64 in
    // BFI alias form; if so, drop down to BFM with the equivalent encoding
    // (BFM Xd, Xn, #0, #63 = BFI Xd, Xn, #0, #64).
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0xDEADBEEFCAFEBABEULL));
                a.mov(x10, Imm(0x1234567890ABCDEFULL));
            },
            [](a64::Assembler& a, uint32_t) {
                // BFM Xd, Xn, #immr=0, #imms=63 → BFI Xd, Xn, #0, #64
                a.bfm(x0, x10, Imm(0), Imm(63));
            });
        snprintf(name, sizeof(name), "BFI variant C (full-width, lsb=0/w=64)");
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_pitfall_tests(const BenchmarkParams& base_params) {
    // Allocate a shared buffer for STNP, misaligned, and CAS tests.
    // 128MB covers all buffer sizes needed.
    constexpr size_t kBufSize = 128ULL * 1024 * 1024;
    void* buf = alloc_pages(kBufSize);
    if (!buf) {
        fprintf(stderr, "run_pitfall_tests: failed to allocate buffer\n");
        return;
    }
    commit_pages(buf, kBufSize);

    run_store_forwarding_tests(base_params);
    run_barrier_tests(base_params);
    run_lrcpc_tests(base_params);
    run_nontemporal_tests(base_params, buf, kBufSize);
    run_misaligned_tests(base_params, buf);
    run_cas_tests(base_params, buf);
    run_bfi_dependency_tests(base_params);

    free_pages(buf, kBufSize);
}

} // namespace arm64bench::gen
