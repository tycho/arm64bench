// gen_memory.cpp
// Memory hierarchy microbenchmark generator.
//
// ── Design notes ─────────────────────────────────────────────────────────────
//
// LATENCY MEASUREMENT (pointer chase)
//   Each node in the chain stores the address of a randomly-chosen next node.
//   The load instruction is:
//       LDR x0, [x0]     — load the 64-bit value at address x0 into x0
//   Since the result is the input to the next iteration, the CPU cannot
//   pipeline or speculate ahead. The measured ns/load is the true cache
//   access latency including tag lookup, data delivery, and register writeback.
//
//   The permutation is generated with Fisher-Yates, seeded from a fixed
//   constant for reproducibility — build_pointer_ring() in gen_common.cpp.
//   The stride between nodes (kNodeStride=256) is chosen to prevent
//   set-conflict aliasing in typical 8-way caches: with 256-byte stride,
//   consecutive nodes never map to the same cache set in a cache whose set
//   count is not a multiple of 4 pages.
//
// BANDWIDTH MEASUREMENT (sequential LDP/STP)
//   The inner loop body issues kBwLines=8 cache lines worth of LDP (load-pair)
//   instructions from consecutive fixed offsets of a single base register:
//       LDP x2, x3, [x0, #0]
//       LDP x4, x5, [x0, #16]
//       ...
//       LDP x8, x9, [x0, #48]   — covers cache line 0 (64 bytes)
//       LDP x2, x3, [x0, #64]
//       ...
//       LDP x8, x9, [x0, #496]  — covers cache line 7 (64 bytes)
//       ADD x0, x0, #512        — advance base (one write, diluted over 32 LDP)
//
//   Because the destination registers (x2..x9) are never read by subsequent
//   loads — they get overwritten each cache line — there is no dependency
//   between loads. The OOO engine can issue all 32 LDP instructions ahead of
//   the ADD, creating up to 32 outstanding cache misses. This is essential
//   for revealing peak bandwidth rather than latency.
//
//   The hardware sequential prefetcher will engage for this pattern, which is
//   intentional: we want to measure the MAXIMUM bandwidth the CPU can deliver
//   to the execution core, which includes prefetcher assistance.
//
// TWO-LEVEL LOOP STRUCTURE
//   Outer loop (x19 = num_passes): counts complete sweeps of the buffer. This
//     is build_loop()'s own counter; the entire inner loop is the loop body,
//     emitted once (unroll = 1).
//   Inner loop (x21 = inner_iters): advances through the buffer in kBwStep
//     increments, executing the LDP/STP block each step.
//
//   After each outer iteration, x0 is reset to buf_base (x20). This keeps
//   the access pattern perfectly sequential and the prefetcher fully engaged
//   across all outer iterations.
//
//   The harness normalizes by: loops=num_passes, instructions_per_loop=cache_lines.
//   This gives min_ns_per_insn = ns_per_cache_line, from which:
//     bandwidth (GB/s) = bytes_per_insn(64) / min_ns_per_insn
//
// PLATFORM MEMORY ALLOCATION
//   alloc_pages() (gen_common.cpp) maps anonymous pages — mmap on POSIX,
//   VirtualAlloc on Windows. A single kMaxBufSize allocation is made at
//   startup. All tests share this backing store; for latency tests the chain
//   links overwrite the memory, but bandwidth tests don't depend on its
//   content (we measure access time, not values).

#include "gen_memory.h"
#include "gen_common.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr size_t kCacheLine  = 64;

// Spacing between pointer-chase nodes. 256 bytes = 4 cache lines.
// Large enough to avoid set-conflict aliasing in typical 8-way caches
// while keeping chains long enough for large buffers.
static constexpr size_t kNodeStride = 256;

// Latency test: loop count is now computed per buffer size by lat_loops_for_size()
// to ensure each sample runs long enough for stable timing. See that function
// for the target-duration rationale.

// Bandwidth test: cache lines per inner step (kBwStep = kBwLines * kCacheLine).
// 8 cache lines per step = 32 LDP instructions, providing 32-way memory-level
// parallelism ahead of the ADD that advances the base pointer.
static constexpr uint32_t kBwLines = 8;
static constexpr size_t   kBwStep  = kBwLines * kCacheLine;  // 512 bytes

// Bandwidth test: target total cache-line accesses per timed call.
// At peak L1 bandwidth (~300 GB/s, 0.213 ns/line): ~10.7ms. Fine.
// At DRAM bandwidth (~20 GB/s, 3.2 ns/line): ~160ms. Acceptable.
static constexpr uint64_t kBwTargetLines = 50'000'000;

// Buffer sizes to sweep — all powers of 2, all multiples of kBwStep (512).
// Covers L1 → L2 → L3 (if present) → DRAM on all target platforms.
//   M1 P-core:   L1D=128KB, L2=12MB (shared cluster), no unified L3
//   Snapdragon X1: L1D=96KB, L2=1.5MB/core, L3=36MB (shared)
static const size_t kBufSizes[] = {
     4ULL*1024,    8ULL*1024,   16ULL*1024,   32ULL*1024,
    64ULL*1024,  128ULL*1024,  256ULL*1024,  512ULL*1024,
     1ULL<<20,    2ULL<<20,     4ULL<<20,
     8ULL<<20,   16ULL<<20,    32ULL<<20,
    64ULL<<20,  128ULL<<20,
};
static constexpr size_t kNumBufSizes = sizeof(kBufSizes) / sizeof(kBufSizes[0]);
static constexpr size_t kMaxBufSize  = 128ULL << 20;  // 128 MB

// ── Latency chase JIT builder ─────────────────────────────────────────────────
//
// Generated loop (pseudo-assembly; the frame is build_loop()'s):
//   sub sp, sp, #48
//   stp x19, x20, [sp] ; stp x21, x22, [sp, #16] ; str x30, [sp, #32]
//   mov x19, #loops
//   mov x0, #chain_head      // 64-bit immediate: MOVZ + up to 3 MOVK
//   align 64
// loop_top:
//   ldr x0, [x0]             // x0 = *(uint64_t*)x0 — the serializing load
//   sub x19, x19, #1
//   cbnz x19, loop_top
//   ...restore x19–x22, x30; add sp, sp, #48...
//   ret x30
//
// Instructions per harness "iteration": 1 (the LDR).
// min_ns_per_insn from the harness = ns/load = cache access latency.

static JitPool::TestFn build_latency_chase(uintptr_t chain_head, uint64_t loops) {
    return build_loop(loops, 1,
        [chain_head](a64::Assembler& a) {
            // Bake chain_head as a 64-bit immediate. AsmJit emits MOVZ + MOVK
            // as needed (1–4 instructions depending on the value).
            a.mov(x0, Imm(static_cast<uint64_t>(chain_head)));
        },
        [](a64::Assembler& a, uint32_t) {
            // The measurement: load the next pointer from the current address.
            // x0 depends on the previous x0, strictly serializing execution.
            a.ldr(x0, ptr(x0));
        });
}

// ── Sequential load bandwidth JIT builder ─────────────────────────────────────
//
// Generated structure (frame, x19 counter and outer loop from build_loop):
//   prologue: save x19–x22, x30
//   mov x19, #num_passes
//   mov x20, #buf_base       // constant (never written inside loop)
//   align 64
// outer_top:
//   mov x0, x20              // reset load pointer to buffer start
//   mov x21, #inner_iters    // inner iteration count = buf_size / kBwStep
// inner_top:
//   ldp x2, x3, [x0, #0]    // ┐
//   ldp x4, x5, [x0, #16]   //  | cache line 0
//   ldp x6, x7, [x0, #32]   //  |
//   ldp x8, x9, [x0, #48]   // ┘
//   ...repeat for 7 more cache lines (offsets 64..496)...
//   add x0, x0, #512         // advance by kBwStep
//   sub x21, x21, #1
//   cbnz x21, inner_top
//   sub x19, x19, #1
//   cbnz x19, outer_top
//   epilogue: restore, ret
//
// The inner loop body contains kBwLines*4 = 32 LDP instructions + 3 control
// instructions. With kBwLines=8, the base pointer x0 is NOT a source for any
// LDP in the current step (all offsets are baked-in immediates), so all 32
// LDP instructions are independent and can be issued simultaneously by the OOO
// engine, creating up to 32 outstanding memory requests.
//
// Harness parameters:
//   loops = num_passes
//   instructions_per_loop = buf_size / kCacheLine  (cache lines per pass)
//   bytes_per_insn = kCacheLine (64)               (for GB/s computation)

static JitPool::TestFn build_seq_load_bw(uintptr_t buf_base, size_t buf_size,
                                          uint64_t num_passes) {
    const uint64_t inner_iters = buf_size / kBwStep;

    return build_loop(num_passes, 1,
        [buf_base](a64::Assembler& a) {
            a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));
        },
        [inner_iters](a64::Assembler& a, uint32_t) {
            a.mov(x0, x20);                     // reset load pointer each pass
            a.mov(x21, Imm(inner_iters));

            Label inner_top = a.new_label();
            a.bind(inner_top);

            // Emit kBwLines cache lines worth of LDP instructions.
            // Destination registers rotate through x2..x9 (4 pairs), repeating
            // each cache line. Reuse is safe: LDP results are never consumed by
            // subsequent LDPs in this block, so there is no dependency chain
            // through destinations.
            for (uint32_t line = 0; line < kBwLines; ++line) {
                const int32_t base_off = static_cast<int32_t>(line * kCacheLine);
                // 4 LDP pairs cover 64 bytes (one cache line).
                a.ldp(x2, x3, ptr(x0, base_off + 0));
                a.ldp(x4, x5, ptr(x0, base_off + 16));
                a.ldp(x6, x7, ptr(x0, base_off + 32));
                a.ldp(x8, x9, ptr(x0, base_off + 48));
            }

            a.add(x0, x0, Imm(static_cast<uint64_t>(kBwStep)));
            a.sub(x21, x21, Imm(1));
            a.cbnz(x21, inner_top);
        });
}

// ── Sequential store bandwidth JIT builder ────────────────────────────────────
//
// Symmetric with the load builder but uses STP instead of LDP.
// The stored value is loaded from x20 (a fixed constant = buf_base address,
// initialized in the prologue). Storing a non-trivial value avoids any
// potential zero-store optimization in the memory subsystem.
//
// STP writes 16 bytes per instruction; 4 STP per cache line (same as LDP).
// On write-allocate caches (the norm on ARM64), each store to a cold line
// will trigger a read-for-ownership to fill the line before writing.
// This means store bandwidth is often limited by both read AND write bus
// capacity — especially visible at DRAM sizes where RFO doubles the traffic.

static JitPool::TestFn build_seq_store_bw(uintptr_t buf_base, size_t buf_size,
                                           uint64_t num_passes) {
    const uint64_t inner_iters = buf_size / kBwStep;

    return build_loop(num_passes, 1,
        [buf_base](a64::Assembler& a) {
            a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));

            // x10 = store value: use buf_base (a non-trivial 64-bit value).
            // Both registers of each STP pair will hold the same value — fine
            // for measuring store bandwidth.
            a.mov(x10, x20);
        },
        [inner_iters](a64::Assembler& a, uint32_t) {
            a.mov(x0, x20);
            a.mov(x21, Imm(inner_iters));

            Label inner_top = a.new_label();
            a.bind(inner_top);

            for (uint32_t line = 0; line < kBwLines; ++line) {
                const int32_t base_off = static_cast<int32_t>(line * kCacheLine);
                a.stp(x10, x10, ptr(x0, base_off + 0));
                a.stp(x10, x10, ptr(x0, base_off + 16));
                a.stp(x10, x10, ptr(x0, base_off + 32));
                a.stp(x10, x10, ptr(x0, base_off + 48));
            }

            a.add(x0, x0, Imm(static_cast<uint64_t>(kBwStep)));
            a.sub(x21, x21, Imm(1));
            a.cbnz(x21, inner_top);
        });
}

// ── Helper: format buffer size as a fixed-width string ───────────────────────

static void format_buf_size(char* out, size_t outlen, size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024)
        snprintf(out, outlen, "%4uGB", static_cast<uint32_t>(bytes / (1024ULL * 1024 * 1024)));
    else if (bytes >= 1024 * 1024)
        snprintf(out, outlen, "%4uMB", static_cast<uint32_t>(bytes / (1024 * 1024)));
    else
        snprintf(out, outlen, "%4uKB", static_cast<uint32_t>(bytes / 1024));
}

// ── Latency sweep ─────────────────────────────────────────────────────────────
//
// Loop count scaling rationale:
//   At L1 latency (~1.5ns/load), a fixed 2M-loop run gives only ~3ms per sample.
//   With the ARM generic timer at 24MHz (~42ns/tick), timer granularity alone
//   contributes ~1.4% error on a 3ms sample, and any OS interrupt during the
//   window dominates the CoV. We need at minimum ~50ms per sample for stable
//   L1 results; 150ms is more comfortable.
//
//   We estimate expected latency conservatively from buffer size and set loops
//   accordingly. The estimates are intentionally pessimistic (slower than real)
//   so that actual sample durations meet or exceed the target. DRAM loops are
//   capped at kLatLoopsMax to prevent excessively long runs.
//
//   Target sample duration: ~150ms.
//     L1  (~2ns/ld):  150ms / 2ns  = 75M loops
//     L2  (~8ns/ld):  150ms / 8ns  = 19M loops  → round to 20M
//     SLC (~35ns/ld): 150ms / 35ns =  4.3M loops → round to 5M
//     DRAM (~110ns/ld): 150ms/110ns = 1.4M loops → round to 2M (cap)

static constexpr uint64_t kLatLoopsMax  = 2'000'000;  // cap for DRAM

static uint64_t lat_loops_for_size(size_t buf_size) {
    // Conservative expected latency per cache level.
    // These are lower bounds — if the real latency is higher, the sample is
    // longer than 150ms, which is fine. If faster, we still get a clean result
    // because we never go below what's needed.
    uint64_t loops;
    if      (buf_size <=  128ULL * 1024)      loops = 75'000'000;   // L1
    else if (buf_size <=    8ULL * 1024*1024) loops = 20'000'000;   // L2/SLC
    else if (buf_size <=   32ULL * 1024*1024) loops =  5'000'000;   // SLC/DRAM edge
    else                                      loops = kLatLoopsMax; // DRAM
    return scale_loops(loops);
}

static void run_latency_sweep(void* buf, const BenchmarkParams& base) {
    char title[96];
    snprintf(title, sizeof(title),
             "Load latency (random pointer chase, %u-byte stride)",
             static_cast<uint32_t>(kNodeStride));
    section(title);

    // Boundary detection thresholds.
    // A latency jump of ≥2× between consecutive buffer sizes almost certainly
    // represents a cache level boundary. A jump of 1.5–2× combined with
    // elevated CoV indicates the buffer is split across two levels (straddling).
    static constexpr double kBoundaryRatio   = 2.0;  // clean jump: fully in new level
    static constexpr double kStraddleRatio   = 1.5;  // partial jump: buffer split
    static constexpr double kStraddleCoV     = 2.0;  // CoV% threshold for straddling

    double prev_min_ns   = 0.0;  // min_ns_per_insn from the previous iteration
    size_t prev_buf_size = 0;

    for (size_t si = 0; si < kNumBufSizes; ++si) {
        const size_t   buf_size = kBufSizes[si];
        const uint64_t loops    = lat_loops_for_size(buf_size);

        void* head = build_pointer_ring(buf, buf_size, kNodeStride);
        if (!head) {
            fprintf(stderr, "  [skipped: build_pointer_ring failed]\n");
            continue;
        }

        JitPool::TestFn fn = build_latency_chase(
            reinterpret_cast<uintptr_t>(head), loops);
        if (!fn) continue;

        const BenchmarkParams p = params_for(base, loops, 1);

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "load latency %s", size_str);

        // Not run_one(): the boundary annotation below needs the result.
        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        // ── Boundary annotation ───────────────────────────────────────────
        // Compare this result against the previous buffer size. Only annotate
        // when there's a meaningful jump — skip the very first result and any
        // result where the latency didn't increase significantly.
        if (prev_min_ns > 0.0 && r.min_ns_per_insn > prev_min_ns) {
            const double ratio = r.min_ns_per_insn / prev_min_ns;

            char prev_str[16];
            format_buf_size(prev_str, sizeof(prev_str), prev_buf_size);

            if (ratio >= kStraddleRatio
                       && r.coeff_variation_pct >= kStraddleCoV) {
                printf("  ↑ straddling cache boundary (%.1f× slower than %s,"
                       " %.1f%% CoV — buffer spans two levels)\n",
                       ratio, prev_str, r.coeff_variation_pct);
            } else if (ratio >= kBoundaryRatio) {
                printf("  ↑ cache level boundary (%.1f× slower than %s)\n",
                       ratio, prev_str);

            }
        }

        prev_min_ns   = r.min_ns_per_insn;
        prev_buf_size = buf_size;
    }
}

// ── TLB hierarchy sweep ───────────────────────────────────────────────────────
//
// Uses the same serializing pointer-chase as the cache latency test, but with
// stride = 4096 bytes (one node per 4KB page). Because each LDR goes to a
// DIFFERENT physical page, every load requires a TLB lookup. With a small page
// count, all entries fit in the L1 DTLB (no miss); beyond that, TLB misses to
// the L2 TLB or hardware page-table walker add measurable latency.
//
// Because each node is only 8 bytes per page, all nodes fit in L1D cache
// when page_count × 8B ≤ L1D (128KB) → page_count ≤ 16384. For page counts
// below this, latency jumps reflect ONLY TLB misses (not cache misses).
//
// Comparison with the existing 256B-stride latency sweep at the same buffer
// size isolates the TLB miss penalty: the difference equals the extra cost of
// taking a TLB miss on every load vs. once every 16 loads (4096/256).
//
// Expected Apple M-series:
//   L1 DTLB:  128 entries (M1) / 192-256 (M2+) — no TLB miss below this count
//   L2 TLB:   ~3072 entries (M1) — additional latency when L1 DTLB misses
//   Page walk: hardware tablewalk, ~5–15 ns penalty on top of L2 TLB latency
//
// Expected Snapdragon X Elite (Oryon):
//   L1 DTLB:  ~64-128 entries — miss threshold at fewer pages than Apple
//   L2 TLB:   ~1024-2048 entries
//   Page walk: hardware MFPT, penalty varies

static void run_tlb_sweep(void* buf, const BenchmarkParams& base) {
    section("TLB hierarchy (pointer chase, 4KB page stride)");

    static constexpr size_t kPageStride = 4096;
    static const uint32_t kPageCounts[] = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    };

    for (uint32_t pages : kPageCounts) {
        const size_t buf_size = static_cast<size_t>(pages) * kPageStride;
        if (buf_size > kMaxBufSize) break;

        void* head = build_pointer_ring(buf, buf_size, kPageStride);
        if (!head) continue;

        const uint64_t loops = lat_loops_for_size(buf_size);
        JitPool::TestFn fn = build_latency_chase(
            reinterpret_cast<uintptr_t>(head), loops);

        char name[80];
        snprintf(name, sizeof(name), "TLB chase  %5u pages (%4zuKB buf)",
                 pages, buf_size / 1024);

        run_one(name, fn, params_for(base, loops, 1));
    }
}

// ── Bandwidth sweep ───────────────────────────────────────────────────────────

static void run_bw_sweep(void* buf, const BenchmarkParams& base, bool is_store) {
    char title[96];
    snprintf(title, sizeof(title),
             "Sequential %s bandwidth (%u-stream %s, %u-byte step)",
             is_store ? "store" : "load",
             kBwLines * 4u,
             is_store ? "STP" : "LDP",
             static_cast<uint32_t>(kBwStep));
    section(title);

    for (size_t si = 0; si < kNumBufSizes; ++si) {
        const size_t buf_size = kBufSizes[si];

        // Compute the number of outer passes that gives ~kBwTargetLines total
        // cache-line accesses. Floor at 4 passes (need at least a few for
        // stable statistics). Ceiling at 2M passes (avoids excessive runtime
        // on trivially small buffers on very fast future cores).
        const uint64_t lines_per_pass = buf_size / kCacheLine;
        uint64_t num_passes = kBwTargetLines / lines_per_pass;
        if (num_passes < 4)          num_passes = 4;
        if (num_passes > 2'000'000)  num_passes = 2'000'000;
        num_passes = scale_loops(num_passes);

        JitPool::TestFn fn = is_store
            ? build_seq_store_bw(reinterpret_cast<uintptr_t>(buf), buf_size, num_passes)
            : build_seq_load_bw (reinterpret_cast<uintptr_t>(buf), buf_size, num_passes);

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "seq %s bw   %s",
                 is_store ? "store" : "load ", size_str);

        run_one(name, fn,
                params_for(base, num_passes,
                           static_cast<uint32_t>(lines_per_pass),
                           static_cast<uint32_t>(kCacheLine)));
    }
}

// ── Non-temporal load bandwidth JIT builder ───────────────────────────────────
//
// Identical to build_seq_load_bw but uses LDNP instead of LDP.
// LDNP is a "non-temporal" load hint: the CPU MAY skip allocating the loaded
// data in the L1/L2 cache (useful for streaming passes that won't reuse data).
//
// Architecture notes:
//   Apple Silicon: LDNP is treated conservatively — the hardware always
//     allocates the load in cache (no bypass). LDNP behaves like LDP.
//     Bandwidth should match LDP for all working set sizes.
//   Qualcomm Oryon / Cortex-A78+: behavior depends on implementation.
//     Some cores bypass L1/L2 for LDNP, which would show lower latency at
//     small sizes (data not cached) but higher bandwidth for DRAM-bound
//     workloads (reduced cache eviction pressure for other active data).
//
// Comparing LDNP vs LDP bandwidth reveals whether the NT hint is honored.

static JitPool::TestFn build_seq_ldnp_bw(uintptr_t buf_base, size_t buf_size,
                                           uint64_t num_passes) {
    const uint64_t inner_iters = buf_size / kBwStep;

    return build_loop(num_passes, 1,
        [buf_base](a64::Assembler& a) {
            a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));
        },
        [inner_iters](a64::Assembler& a, uint32_t) {
            a.mov(x0, x20);
            a.mov(x21, Imm(inner_iters));

            Label inner_top = a.new_label();
            a.bind(inner_top);

            for (uint32_t line = 0; line < kBwLines; ++line) {
                const int32_t base_off = static_cast<int32_t>(line * kCacheLine);
                a.ldnp(x2, x3, ptr(x0, base_off + 0));
                a.ldnp(x4, x5, ptr(x0, base_off + 16));
                a.ldnp(x6, x7, ptr(x0, base_off + 32));
                a.ldnp(x8, x9, ptr(x0, base_off + 48));
            }

            a.add(x0, x0, Imm(static_cast<uint64_t>(kBwStep)));
            a.sub(x21, x21, Imm(1));
            a.cbnz(x21, inner_top);
        });
}

// ── LDP→STP copy bandwidth JIT builder ───────────────────────────────────────
//
// Copies from a source buffer to a destination buffer using LDP/STP pairs.
// Both buffers advance in lockstep; the total working set is 2×buf_size.
//
// Register layout:
//   x19 = outer pass counter (build_loop's; not touched here)
//   x20 = src_base (constant)
//   x21 = inner iteration counter
//   x22 = dst_base (constant; callee-saved so it persists across inner loops)
//   x0  = src read pointer (reset to x20 each outer pass)
//   x11 = dst write pointer (reset to x22 each outer pass; caller-saved)
//
// The LDP reads into x2..x9 (4 pairs = 64 bytes = 1 cache line); STP
// immediately writes those pairs to the destination. The load→store pair
// has a data dependency that may prevent full pipelining — this reflects
// realistic memcpy behavior, not maximum theoretical bandwidth.

static JitPool::TestFn build_seq_copy_bw(uintptr_t src_base, uintptr_t dst_base,
                                           size_t buf_size, uint64_t num_passes) {
    const uint64_t inner_iters = buf_size / kBwStep;

    return build_loop(num_passes, 1,
        [src_base, dst_base](a64::Assembler& a) {
            a.mov(x20, Imm(static_cast<uint64_t>(src_base)));
            a.mov(x22, Imm(static_cast<uint64_t>(dst_base)));
        },
        [inner_iters](a64::Assembler& a, uint32_t) {
            a.mov(x0,  x20);           // src read pointer
            a.mov(x11, x22);           // dst write pointer (x11 = caller-saved)
            a.mov(x21, Imm(inner_iters));

            Label inner_top = a.new_label();
            a.bind(inner_top);

            // Copy kBwLines cache lines per step.
            // Each cache line: 4 LDP pairs (read) → 4 STP pairs (write).
            for (uint32_t line = 0; line < kBwLines; ++line) {
                const int32_t base_off = static_cast<int32_t>(line * kCacheLine);
                a.ldp(x2, x3, ptr(x0,  base_off + 0));
                a.stp(x2, x3, ptr(x11, base_off + 0));
                a.ldp(x4, x5, ptr(x0,  base_off + 16));
                a.stp(x4, x5, ptr(x11, base_off + 16));
                a.ldp(x6, x7, ptr(x0,  base_off + 32));
                a.stp(x6, x7, ptr(x11, base_off + 32));
                a.ldp(x8, x9, ptr(x0,  base_off + 48));
                a.stp(x8, x9, ptr(x11, base_off + 48));
            }

            a.add(x0,  x0,  Imm(static_cast<uint64_t>(kBwStep)));
            a.add(x11, x11, Imm(static_cast<uint64_t>(kBwStep)));
            a.sub(x21, x21, Imm(1));
            a.cbnz(x21, inner_top);
        });
}

// ── Non-temporal load bandwidth sweep ────────────────────────────────────────

static void run_ldnp_bw_sweep(void* buf, const BenchmarkParams& base) {
    char title[96];
    snprintf(title, sizeof(title),
             "Non-temporal load bandwidth (LDNP, %u-stream, %u-byte step)",
             kBwLines * 4u, static_cast<uint32_t>(kBwStep));
    section(title);

    for (size_t si = 0; si < kNumBufSizes; ++si) {
        const size_t buf_size = kBufSizes[si];

        const uint64_t lines_per_pass = buf_size / kCacheLine;
        uint64_t num_passes = kBwTargetLines / lines_per_pass;
        if (num_passes < 4)          num_passes = 4;
        if (num_passes > 2'000'000)  num_passes = 2'000'000;
        num_passes = scale_loops(num_passes);

        JitPool::TestFn fn = build_seq_ldnp_bw(
            reinterpret_cast<uintptr_t>(buf), buf_size, num_passes);

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "LDNP load bw   %s", size_str);

        run_one(name, fn,
                params_for(base, num_passes,
                           static_cast<uint32_t>(lines_per_pass),
                           static_cast<uint32_t>(kCacheLine)));
    }
}

// ── LDP→STP copy bandwidth sweep ─────────────────────────────────────────────
//
// Note: the total working set is 2×buf_size (src + dst). To keep src and dst
// within the allocated buffer, we limit buf_size to kMaxBufSize/2.

static void run_copy_bw_sweep(void* buf, const BenchmarkParams& base) {
    char title[96];
    snprintf(title, sizeof(title),
             "LDP→STP copy bandwidth (%u-stream, %u-byte step)",
             kBwLines * 4u, static_cast<uint32_t>(kBwStep));
    section(title);

    const uintptr_t buf_addr = reinterpret_cast<uintptr_t>(buf);
    const size_t    half     = kMaxBufSize / 2;

    for (size_t si = 0; si < kNumBufSizes; ++si) {
        const size_t buf_size = kBufSizes[si];
        if (buf_size > half) break;  // total working set would exceed allocation

        const uintptr_t src_base = buf_addr;
        const uintptr_t dst_base = buf_addr + half;

        const uint64_t lines_per_pass = buf_size / kCacheLine;
        uint64_t num_passes = kBwTargetLines / lines_per_pass;
        if (num_passes < 4)          num_passes = 4;
        if (num_passes > 2'000'000)  num_passes = 2'000'000;
        num_passes = scale_loops(num_passes);

        JitPool::TestFn fn = build_seq_copy_bw(src_base, dst_base, buf_size, num_passes);

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "LDP→STP copy   %s", size_str);

        run_one(name, fn,
                params_for(base, num_passes,
                           static_cast<uint32_t>(lines_per_pass),
                           static_cast<uint32_t>(kCacheLine)));
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_memory_tests(const BenchmarkParams& base_params) {
    // Allocate the backing buffer once. All sweeps share it.
    void* buf = alloc_pages(kMaxBufSize);
    if (!buf) {
        fprintf(stderr, "run_memory_tests: failed to allocate %zuMB backing buffer\n",
                kMaxBufSize / (1024 * 1024));
        return;
    }

    // Touch all pages to fault them in before benchmarking. Without this,
    // the first sweep would pay page-fault overhead on top of cache-miss
    // latency, conflating two completely different costs.
    // commit_pages() deliberately avoids memset — a sequential write over the
    // whole buffer would warm the cache and influence the first latency test.
    // It writes a single byte per 4KB page instead.
    commit_pages(buf, kMaxBufSize);

    run_latency_sweep(buf, base_params);
    run_tlb_sweep(buf, base_params);
    run_bw_sweep(buf, base_params, /*is_store=*/false);
    run_bw_sweep(buf, base_params, /*is_store=*/true);
    run_ldnp_bw_sweep(buf, base_params);
    run_copy_bw_sweep(buf, base_params);

    free_pages(buf, kMaxBufSize);
}

} // namespace arm64bench::gen
