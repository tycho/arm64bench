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
#include "jit_buffer.h"
#include "harness.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstdlib>   // malloc, free

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Platform memory helpers ───────────────────────────────────────────────────

static void* alloc_pages(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

static void free_pages(void* p, size_t size) {
    if (!p) return;
#if defined(_WIN32)
    (void)size; VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, size);
#endif
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

static void run_one(const char* name, JitPool::TestFn fn,
                    const BenchmarkParams& p) {
    if (!fn) return;
    benchmark(fn, name, p);
    g_jit_pool->release(fn);
}

static BenchmarkParams make_lat_params(const BenchmarkParams& base,
                                       uint64_t loops, uint32_t unroll = 8) {
    BenchmarkParams p       = base;
    p.loops                 = loops;
    p.instructions_per_loop = unroll;
    p.bytes_per_insn        = 0;
    return p;
}

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
//   Using [sp-8] as the store/load address — this is "red zone" territory
//   on ARM64 (unlike x86-64, ARM64 ABI doesn't define a red zone, but we
//   can safely use addresses below sp as a temporary scratch slot since no
//   signal handlers or interrupts will disturb user-level code in practice).
//   We don't need to allocate stack space for a scratch slot.
//
//   RATIONALE FOR BELOW-SP: The store/load slot must be at a known fixed
//   address that is a valid data address. We use ptr(sp, -8) to avoid
//   allocating additional stack space in the prologue while keeping the
//   address data-cache-resident. This is benign for benchmarking purposes
//   even though it's technically below the stack pointer — interrupts may
//   briefly corrupt it, but the STORE immediately before each LOAD means
//   the value is always freshly written before it's read.

// Forward declaration.
static void run_store_forwarding_tests(const BenchmarkParams& base);
static void run_barrier_tests(const BenchmarkParams& base);
static void run_nontemporal_tests(const BenchmarkParams& base, void* buf, size_t bufsz);
static void run_misaligned_tests(const BenchmarkParams& base, void* buf);
static void run_cas_tests(const BenchmarkParams& base, void* buf);

// ── STL forwarding loop builder ───────────────────────────────────────────────
//
// Each outer iteration:
//   [setup]: x0 = current value, x9 = scratch address (below sp)
//   store:   STR/STRW/STRH/STRB x0|w0, [x9, #store_offset]
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
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Prologue: save x19 (loop counter), x30 (LR).
    // x9 = scratch address slot just below sp, established once.
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x30, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x0,  Imm(0x0102030405060708ULL));  // non-trivial initial value
    // x9 points to a 16-byte slot at [sp-16]. We sub sp a second time to
    // give ourselves a clean 16-byte slot, keeping sp 16-byte aligned.
    a.sub(sp, sp, Imm(16));
    a.mov(x9, sp);   // x9 = scratch buffer address

    a.align(AlignMode::kCode, 64);
    Label loop_top = a.new_label();
    a.bind(loop_top);

    for (uint32_t u = 0; u < unroll; ++u) {
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
    }

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    // Restore extra frame.
    a.add(sp, sp, Imm(16));
    a.ldp(x19, x30, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

static void run_store_forwarding_tests(const BenchmarkParams& base) {
    printf("\n── Store-to-load forwarding ────────────────────────────────────\n");
    printf("  clk/insn = latency of one STORE+LOAD pair (unroll=8).\n"
           "  M1 penalty for mismatch: ~+8–12 clk vs matched case.\n\n");

    // Loop counts: forwarding latency ≈ 4–16 cycles → target ~100ms.
    // At 15 cyc × (1/3.2 GHz) × 8 unroll = ~37.5ns/iter → 100ms/37.5ns ≈ 2.7M.
    const uint64_t loops  = 3'000'000;
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
        run_one(name, fn, make_lat_params(base, loops, unroll));
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
    printf("\n── Memory ordering barriers ────────────────────────────────────\n");
    printf("  clk/insn = cycles per barrier instruction (unroll=8).\n\n");

    const uint64_t loops  = 3'000'000;
    const uint32_t unroll = 8;
    char name[80];

    auto make_p = [&](uint64_t l, uint32_t u) {
        return make_lat_params(base, l, u);
    };

    // ── Shuffled L1 pointer ring ───────────────────────────────────────────
    // Build a 64-node ring (512 bytes, fits in L1) with a Fisher-Yates shuffle
    // so that every load returns a *different* address. Placed on the stack so
    // it stays alive for the entire run_barrier_tests call.
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
    {
        uint32_t perm[kRingN];
        for (uint32_t i = 0; i < kRingN; ++i) perm[i] = i;
        uint64_t rng = 0xDEADBEEF12345678ULL;
        auto xs = [&]() -> uint64_t {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
        };
        for (uint32_t i = kRingN - 1; i > 0; --i) {
            uint32_t j = xs() % (i + 1);
            uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for (uint32_t i = 0; i < kRingN; ++i)
            ring_buf[perm[i]] = reinterpret_cast<uintptr_t>(
                                    &ring_buf[perm[(i + 1) % kRingN]]);
    }
    const uintptr_t ring_head = reinterpret_cast<uintptr_t>(&ring_buf[0]);

    // ── LDR baseline (plain load, no ordering) ────────────────────────────
    // Pointer-chase through the shuffled L1 ring. This is the true L1 load
    // latency reference; a barrier adds overhead on top of this.
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x0, Imm(static_cast<uint64_t>(ring_head)));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u)
                a.ldr(x0, ptr(x0));
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "LDR x64 (L1 chain, baseline)");
        run_one(name, fn, make_p(loops, unroll));
    }

    // ── LDAR: load-acquire ────────────────────────────────────────────────
    // Same ring, LDAR instead of LDR. LDAR prevents reordering of later
    // accesses before this load and inhibits load value speculation.
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x0, Imm(static_cast<uint64_t>(ring_head)));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u)
                a.ldar(x0, ptr(x0));
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "LDAR x64 (load-acquire)");
        run_one(name, fn, make_p(loops, unroll));
    }

    // ── DMB ISH ───────────────────────────────────────────────────────────
    // A stream of DMB ISH instructions with no surrounding loads/stores.
    // This measures the raw barrier serialization cost.
    // In practice a DMB always occurs between memory accesses, so this is
    // a lower bound on the cost it adds to lock/unlock operations.
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u)
                a.dmb(Imm(Predicate::DB::kISH));
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "DMB ISH");
        run_one(name, fn, make_p(loops, unroll));
    }

    // ── DSB ISH ───────────────────────────────────────────────────────────
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u)
                a.dsb(Imm(Predicate::DB::kISH));
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "DSB ISH");
        run_one(name, fn, make_p(loops, unroll));
    }

    // ── ISB (Instruction Synchronization Barrier) ─────────────────────────
    // Flushes and refills the instruction pipeline. The most expensive
    // barrier — used only when instruction cache coherency is required
    // (e.g. after writing JIT code into executable memory).
    // We expect this to be substantially more expensive than DMB/DSB.
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u)
                a.isb(Imm(0xF));  // 0xF = SY option
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "ISB SY");
        run_one(name, fn, make_p(loops, unroll));
    }

    // ── LDR + DMB ISH (acquire pattern) vs LDAR ───────────────────────────
    // The traditional way to implement a load-acquire is LDR followed by DMB.
    // LDAR is the preferred single-instruction equivalent.
    // If LDR+DMB total cost ≈ LDAR cost, the hardware is folding them.
    // If LDR+DMB is more expensive, LDAR is genuinely faster.
    // Uses the same shuffled L1 ring as the LDR baseline.
    {
        const uint32_t pair_unroll = 4;  // 4 pairs = 8 instructions
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x0, Imm(static_cast<uint64_t>(ring_head)));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < pair_unroll; ++u) {
                a.ldr(x0, ptr(x0));
                a.dmb(Imm(Predicate::DB::kISH));
            }
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "LDR + DMB ISH (manual acquire, 4 pairs)");
        run_one(name, fn, make_p(loops, pair_unroll * 2));
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
    CodeHolder code; g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(32));
    a.stp(x19, x20, ptr(sp));
    a.str(x21, ptr(sp, 16));

    a.mov(x19, Imm(num_passes));
    a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));
    a.mov(x10, x20);  // store value (non-trivial = buf_base)

    a.align(AlignMode::kCode, 64);
    Label outer = a.new_label(), inner = a.new_label();
    a.bind(outer);
    a.mov(x0,  x20);
    a.mov(x21, Imm(inner_iters));
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

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, outer);

    a.ldp(x19, x20, ptr(sp));
    a.ldr(x21, ptr(sp, 16));
    a.add(sp, sp, Imm(32));
    a.ret(x30);
    return g_jit_pool->compile(code);
}

static void run_nontemporal_tests(const BenchmarkParams& base,
                                  void* buf, size_t bufsz) {
    printf("\n── Non-temporal stores (STNP vs STP) ──────────────────────────\n");
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
        const uint64_t p_clamped = (passes < 4) ? 4 : (passes > 2'000'000 ? 2'000'000 : passes);

        BenchmarkParams p       = base;
        p.loops                 = p_clamped;
        p.instructions_per_loop = static_cast<uint32_t>(lines);
        p.bytes_per_insn        = static_cast<uint32_t>(kNTCacheLine);

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
    printf("\n── Misaligned load latency ─────────────────────────────────────\n");
    printf("  Pointer-chase through a buffer; each pointer is misaligned\n"
           "  by the given byte offset from 8-byte alignment.\n\n");

    const uint64_t loops   = 10'000'000;
    char name[80];

    // Offsets to test. 0 = naturally aligned. Others probe crossing points.
    const int32_t offsets[] = { 0, 1, 4, 7, 56, 60, 63 };

    for (int32_t off : offsets) {
        // Build a pointer chain in the buffer with each node at
        // (naturally_aligned_addr + off). The chain must stay within buf.
        // Use a simple stride of 256 bytes between nodes.
        const size_t stride    = 256;
        const size_t buf_size  = 2ULL * 1024 * 1024;  // 2MB — fits in L2
        const size_t n_nodes   = buf_size / stride;

        uint8_t* base_ptr = static_cast<uint8_t*>(buf);

        // Build permuted chain with Fisher-Yates (fixed seed).
        uint32_t* perm = static_cast<uint32_t*>(malloc(n_nodes * sizeof(uint32_t)));
        if (!perm) continue;
        for (uint32_t i = 0; i < n_nodes; ++i) perm[i] = i;
        uint64_t rng = 0xDEADBEEF12345678ULL;
        auto xorshift = [&]() -> uint64_t {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
        };
        for (size_t i = n_nodes - 1; i > 0; --i) {
            size_t j = xorshift() % (i + 1);
            uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }

        // Write chain links: at each slot, store the address of the next slot.
        // Each "slot" is at (perm[i] * stride + off) bytes from base.
        for (size_t i = 0; i < n_nodes; ++i) {
            void** slot = reinterpret_cast<void**>(
                base_ptr + static_cast<size_t>(perm[i]) * stride + off);
            *slot = base_ptr + static_cast<size_t>(perm[(i + 1) % n_nodes]) * stride + off;
        }
        uintptr_t head = reinterpret_cast<uintptr_t>(
            base_ptr + static_cast<size_t>(perm[0]) * stride + off);
        free(perm);

        // Build the pointer-chase JIT loop.
        CodeHolder code; g_jit_pool->init_code_holder(code);
        a64::Assembler a(&code);
        a.sub(sp, sp, Imm(16));
        a.stp(x19, x30, ptr(sp));
        a.mov(x19, Imm(loops));
        a.mov(x0, Imm(static_cast<uint64_t>(head)));
        a.align(AlignMode::kCode, 64);
        Label top = a.new_label(); a.bind(top);
        a.ldr(x0, ptr(x0));
        a.sub(x19, x19, Imm(1));
        a.cbnz(x19, top);
        a.ldp(x19, x30, ptr(sp));
        a.add(sp, sp, Imm(16));
        a.ret(x30);
        auto fn = g_jit_pool->compile(code);

        const char* boundary = (off == 0)    ? "(aligned)"        :
                               (off < 8)     ? "(within 8B word)" :
                               (off <= 55)   ? "(within cache line)" :
                               (off <= 63)   ? "(crosses cache line)" : "";
        snprintf(name, sizeof(name), "LDR misalign +%2d bytes %s", off, boundary);
        run_one(name, fn, make_lat_params(base, loops, 1));
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
    printf("\n── CAS (Compare-and-Swap) latency ──────────────────────────────\n");
    printf("  Single-threaded CAS on an L1-resident cache line.\n"
           "  clk/insn = total CAS round-trip latency (load+compare+store).\n\n");

    const uint64_t loops  = 5'000'000;
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
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
            a.mov(x2, Imm(0));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u) {
                a.mov(x1, Imm(0));
                a.cas(x1, x2, ptr(x9));
            }
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "CAS   x64 relaxed (always succeeds)");
        run_one(name, fn, make_lat_params(base, loops, unroll));
    }

    // ── CASAL (acquire+release) ───────────────────────────────────────────
    // Full sequential-consistency CAS. This is what a correct spinlock
    // acquire needs. Its latency is the minimum uncontended lock cycle time.
    {
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
            a.mov(x2, Imm(0));
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u) {
                a.mov(x1, Imm(0));
                a.casal(x1, x2, ptr(x9));
            }
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "CASAL x64 acq+rel (spinlock acquire)");
        run_one(name, fn, make_lat_params(base, loops, unroll));
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
        auto fn = [&] {
            CodeHolder code; g_jit_pool->init_code_holder(code);
            a64::Assembler a(&code);
            a.sub(sp, sp, Imm(16));
            a.stp(x19, x30, ptr(sp));
            a.mov(x19, Imm(loops));
            a.mov(x9, Imm(static_cast<uint64_t>(cas_addr)));
            a.mov(x0, Imm(0));   // new value = 0
            a.align(AlignMode::kCode, 64);
            Label top = a.new_label(); a.bind(top);
            for (uint32_t u = 0; u < unroll; ++u) {
                a.ldaxr(x1, ptr(x9));      // load-acquire-exclusive: x1 = [x9]
                a.stlxr(w2, x0, ptr(x9)); // store-release-exclusive: [x9] = 0
                // w2 = 0 on success, 1 on failure. We don't check or retry.
                // The LDAXR→STLXR window contains zero other instructions,
                // minimising the chance of preemption breaking the reservation.
            }
            a.sub(x19, x19, Imm(1));
            a.cbnz(x19, top);
            a.ldp(x19, x30, ptr(sp));
            a.add(sp, sp, Imm(16));
            a.ret(x30);
            return g_jit_pool->compile(code);
        }();
        snprintf(name, sizeof(name), "LDAXR+STLXR (LL/SC, no-retry)");
        run_one(name, fn, make_lat_params(base, loops, unroll));
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
    // Touch all pages to commit them.
    { uint8_t* p = static_cast<uint8_t*>(buf);
      for (size_t off = 0; off < kBufSize; off += 4096) p[off] = 0; }

    run_store_forwarding_tests(base_params);
    run_barrier_tests(base_params);
    run_nontemporal_tests(base_params, buf, kBufSize);
    run_misaligned_tests(base_params, buf);
    run_cas_tests(base_params, buf);

    free_pages(buf, kBufSize);
}

} // namespace arm64bench::gen
