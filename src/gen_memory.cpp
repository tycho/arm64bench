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
//   constant for reproducibility. The stride between nodes (kNodeStride=256)
//   is chosen to prevent set-conflict aliasing in typical 8-way caches:
//   with 256-byte stride, consecutive nodes never map to the same cache set
//   in a cache whose set count is not a multiple of 4 pages.
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
//   Outer loop (x19 = num_passes): counts complete sweeps of the buffer.
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
//   Uses mmap(MAP_ANONYMOUS|MAP_PRIVATE) on POSIX and VirtualAlloc on Windows.
//   A single kMaxBufSize allocation is made at startup. All tests share this
//   backing store; for latency tests the chain links overwrite the memory,
//   but bandwidth tests don't depend on its content (we measure access time,
//   not values).

#include "gen_memory.h"
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

// ── Register lookup table ─────────────────────────────────────────────────────
// (Same pattern as gen_integer.cpp — x0–x15 are scratch, x19–x21 reserved.)

static const a64::Gp kXRegs[] = {
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
};
static inline const a64::Gp& xr(uint32_t i) { return kXRegs[i]; }

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr size_t kCacheLine  = 64;

// Spacing between pointer-chase nodes. 256 bytes = 4 cache lines.
// Large enough to avoid set-conflict aliasing in typical 8-way caches
// while keeping chains long enough for large buffers.
static constexpr size_t kNodeStride = 256;

// Latency test: iterations of the pointer-chase loop per timed call.
// At worst-case DRAM latency (~100 ns/load) this gives ~200ms per sample.
// At L1 latency (~1.3 ns/load) it gives ~2.6ms — enough for tick-alignment.
static constexpr uint64_t kLatLoops = 2'000'000;

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

// ── Platform memory allocation ────────────────────────────────────────────────
//
// We need large anonymous allocations that don't interact with malloc's heap
// management or trigger any memory-accounting callbacks. mmap and VirtualAlloc
// both provide zero-initialized, demand-paged memory.

static void* alloc_large(size_t size) {
#if defined(_WIN32)
    void* p = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return p;
#else
    void* p = mmap(nullptr, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

static void free_large(void* p, size_t size) {
    if (!p) return;
#if defined(_WIN32)
    (void)size;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, size);
#endif
}

// ── Pseudo-random number generator ───────────────────────────────────────────
// xorshift64: fast, statistically adequate for shuffling indices.
// Fixed seed ensures the pointer chain is identical across runs
// (important for reproducibility — different chain orders can have
// measurably different TLB pressures for very large buffers).

static uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// ── Pointer chain setup ───────────────────────────────────────────────────────
//
// Divides buf[0..size) into n_nodes = size/stride slots.
// Generates a random cyclic permutation of all n_nodes slots using
// Fisher-Yates, then writes at each slot the address of the next slot
// in the permuted order.
//
// The result is a single closed cycle visiting every slot exactly once.
// Starting from the returned head pointer and following LDR x0,[x0]
// repeatedly will visit every slot before returning to the start.
//
// Returns the starting address (head of the chain).

static void* setup_pointer_chase(void* buf, size_t size, size_t stride) {
    const size_t n_nodes = size / stride;
    if (n_nodes < 2) return nullptr;

    // Allocate a temporary index array. Using malloc here is fine — this
    // is one-time setup outside the timed region. For 128MB with 256B stride:
    // n_nodes = 512K, array = 512K * 4B = 2MB. Acceptable.
    uint32_t* perm = static_cast<uint32_t*>(malloc(n_nodes * sizeof(uint32_t)));
    if (!perm) {
        fprintf(stderr, "setup_pointer_chase: malloc(%zu) failed\n",
                n_nodes * sizeof(uint32_t));
        return nullptr;
    }

    // Initialize to identity permutation.
    for (uint32_t i = 0; i < n_nodes; ++i)
        perm[i] = i;

    // Fisher-Yates shuffle.
    uint64_t rng = 0xDEADBEEF12345678ULL;  // fixed seed for reproducibility
    for (size_t i = n_nodes - 1; i > 0; --i) {
        const size_t j = xorshift64(rng) % (i + 1);
        const uint32_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    // Write chain links: at slot perm[i], store the address of slot perm[i+1].
    uint8_t* const base = static_cast<uint8_t*>(buf);
    for (size_t i = 0; i < n_nodes; ++i) {
        void** slot = reinterpret_cast<void**>(base + perm[i] * stride);
        *slot = base + perm[(i + 1) % n_nodes] * stride;
    }

    void* head = base + perm[0] * stride;
    free(perm);
    return head;
}

// ── Latency chase JIT builder ─────────────────────────────────────────────────
//
// Generated loop (pseudo-assembly):
//   sub sp, sp, #16
//   str x19, [sp]
//   mov x19, #loops
//   mov x0, #chain_head      // 64-bit immediate: MOVZ + up to 3 MOVK
//   align 64
// loop_top:
//   ldr x0, [x0]             // x0 = *(uint64_t*)x0 — the serializing load
//   sub x19, x19, #1
//   cbnz x19, loop_top
//   ldr x19, [sp]
//   add sp, sp, #16
//   ret x30
//
// Instructions per harness "iteration": 1 (the LDR).
// min_ns_per_insn from the harness = ns/load = cache access latency.

static JitPool::TestFn build_latency_chase(uintptr_t chain_head, uint64_t loops) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Prologue: save only x19 (loop counter).
    // Use 16-byte frame to maintain ABI stack alignment.
    a.sub(sp, sp, Imm(16));
    a.str(x19, ptr(sp));

    a.mov(x19, Imm(loops));

    // Bake chain_head as a 64-bit immediate. AsmJit emits MOVZ + MOVK as
    // needed (1–4 instructions depending on the value).
    a.mov(x0, Imm(static_cast<uint64_t>(chain_head)));

    // Align to cache line boundary: prevents the loop from spanning two
    // fetch groups, which would cause unpredictable front-end stalls.
    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // The measurement: load the next pointer from the current address.
    // x0 depends on the previous x0, strictly serializing execution.
    a.ldr(x0, ptr(x0));

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    // Epilogue.
    a.ldr(x19, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Sequential load bandwidth JIT builder ─────────────────────────────────────
//
// Generated structure:
//   prologue: save x19, x20, x21
//   mov x19, #num_passes
//   mov x20, #buf_base       // constant (never written inside loop)
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

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Prologue: save x19 (outer counter), x20 (buf_base constant), x21 (inner counter).
    // 32-byte frame: stp x19,x20 at [sp], str x21 at [sp+16].
    a.sub(sp, sp, Imm(32));
    a.stp(x19, x20, ptr(sp));
    a.str(x21, ptr(sp, 16));

    a.mov(x19, Imm(num_passes));
    a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));

    a.align(AlignMode::kCode, 64);

    Label outer_top = a.new_label();
    Label inner_top = a.new_label();

    a.bind(outer_top);
    a.mov(x0, x20);                         // reset load pointer each pass
    a.mov(x21, Imm(inner_iters));

    a.bind(inner_top);

    // Emit kBwLines cache lines worth of LDP instructions.
    // Destination registers rotate through x2..x9 (4 pairs), repeating each
    // cache line. Reuse is safe: LDP results are never consumed by subsequent
    // LDPs in this block, so there is no dependency chain through destinations.
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

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, outer_top);

    // Epilogue.
    a.ldp(x19, x20, ptr(sp));
    a.ldr(x21, ptr(sp, 16));
    a.add(sp, sp, Imm(32));
    a.ret(x30);

    return g_jit_pool->compile(code);
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

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(32));
    a.stp(x19, x20, ptr(sp));
    a.str(x21, ptr(sp, 16));

    a.mov(x19, Imm(num_passes));
    a.mov(x20, Imm(static_cast<uint64_t>(buf_base)));

    // x10 = store value: use buf_base (a non-trivial 64-bit value).
    // Both registers of each STP pair will hold the same value — fine for
    // measuring store bandwidth.
    a.mov(x10, x20);

    a.align(AlignMode::kCode, 64);

    Label outer_top = a.new_label();
    Label inner_top = a.new_label();

    a.bind(outer_top);
    a.mov(x0, x20);
    a.mov(x21, Imm(inner_iters));

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

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, outer_top);

    a.ldp(x19, x20, ptr(sp));
    a.ldr(x21, ptr(sp, 16));
    a.add(sp, sp, Imm(32));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Helper: format buffer size as a fixed-width string ───────────────────────

static void format_buf_size(char* out, size_t outlen, size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024)
        snprintf(out, outlen, "%4zuGB", bytes / (1024ULL * 1024 * 1024));
    else if (bytes >= 1024 * 1024)
        snprintf(out, outlen, "%4zuMB", bytes / (1024 * 1024));
    else
        snprintf(out, outlen, "%4zuKB", bytes / 1024);
}

// ── Latency sweep ─────────────────────────────────────────────────────────────

static void run_latency_sweep(void* buf, const BenchmarkParams& base) {
    printf("\n── Load latency (random pointer chase, %zu-byte stride) ──────────\n",
           kNodeStride);

    BenchmarkParams p = base;
    p.loops               = kLatLoops;
    p.instructions_per_loop = 1;    // one LDR per iteration
    p.bytes_per_insn      = 0;      // not a bandwidth test
    // Fewer warmup/samples for latency: DRAM runs can be slow (200ms/sample).
    // Use the base values but allow callers to override via base_params.

    for (size_t si = 0; si < kNumBufSizes; ++si) {
        const size_t buf_size = kBufSizes[si];

        // Set up the random pointer chain within the first buf_size bytes.
        void* head = setup_pointer_chase(buf, buf_size, kNodeStride);
        if (!head) {
            fprintf(stderr, "  [skipped: setup_pointer_chase failed for %zuKB]\n",
                    buf_size / 1024);
            continue;
        }

        JitPool::TestFn fn = build_latency_chase(
            reinterpret_cast<uintptr_t>(head), kLatLoops);
        if (!fn) continue;

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "load latency %s", size_str);

        benchmark(fn, name, p);
        g_jit_pool->release(fn);
    }
}

// ── Bandwidth sweep ───────────────────────────────────────────────────────────

static void run_bw_sweep(void* buf, const BenchmarkParams& base, bool is_store) {
    printf("\n── Sequential %s bandwidth (%u-stream %s, %zu-byte step) ─────────\n",
           is_store ? "store" : "load",
           kBwLines * 4u,  // LDP/STP count per step (4 per cache line × kBwLines)
           is_store ? "STP" : "LDP",
           kBwStep);

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

        JitPool::TestFn fn = is_store
            ? build_seq_store_bw(reinterpret_cast<uintptr_t>(buf), buf_size, num_passes)
            : build_seq_load_bw (reinterpret_cast<uintptr_t>(buf), buf_size, num_passes);
        if (!fn) continue;

        BenchmarkParams p         = base;
        p.loops                   = num_passes;
        p.instructions_per_loop   = static_cast<uint32_t>(lines_per_pass);
        p.bytes_per_insn          = static_cast<uint32_t>(kCacheLine);

        char size_str[16];
        format_buf_size(size_str, sizeof(size_str), buf_size);

        char name[64];
        snprintf(name, sizeof(name), "seq %s bw   %s",
                 is_store ? "store" : "load ", size_str);

        benchmark(fn, name, p);
        g_jit_pool->release(fn);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_memory_tests(const BenchmarkParams& base_params) {
    // Allocate the backing buffer once. All sweeps share it.
    void* buf = alloc_large(kMaxBufSize);
    if (!buf) {
        fprintf(stderr, "run_memory_tests: failed to allocate %zuMB backing buffer\n",
                kMaxBufSize / (1024 * 1024));
        return;
    }

    // Touch all pages to fault them in before benchmarking. Without this,
    // the first sweep would pay page-fault overhead on top of cache-miss
    // latency, conflating two completely different costs.
    // memset is deliberately avoided here — we don't want to warm the cache
    // with a sequential write that might influence the first latency test.
    // Instead, write a single byte per page (4KB) to commit the pages.
    {
        uint8_t* p = static_cast<uint8_t*>(buf);
        for (size_t off = 0; off < kMaxBufSize; off += 4096)
            p[off] = 0;
    }

    run_latency_sweep(buf, base_params);
    run_bw_sweep(buf, base_params, /*is_store=*/false);
    run_bw_sweep(buf, base_params, /*is_store=*/true);

    free_large(buf, kMaxBufSize);
}

} // namespace arm64bench::gen
