// gen_common.cpp
// Non-template parts of the shared generator scaffolding. See gen_common.h.

#include "gen_common.h"

#include <cstdlib>
#include <cstring>

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

// ── Register tables ───────────────────────────────────────────────────────────

const Gp& xr(uint32_t i) {
    static const Gp kX[16] = {
        x0, x1, x2,  x3,  x4,  x5,  x6,  x7,
        x8, x9, x10, x11, x12, x13, x14, x15,
    };
    return kX[i & 15u];
}

const Gp& wr(uint32_t i) {
    static const Gp kW[16] = {
        w0, w1, w2,  w3,  w4,  w5,  w6,  w7,
        w8, w9, w10, w11, w12, w13, w14, w15,
    };
    return kW[i & 15u];
}

// ── Benchmark-run helpers ─────────────────────────────────────────────────────

BenchmarkParams params_for(const BenchmarkParams& base, uint64_t loops,
                           uint32_t insns_per_loop, uint32_t bytes_per_insn) {
    BenchmarkParams p       = base;
    p.loops                 = loops;
    p.instructions_per_loop = insns_per_loop;
    p.bytes_per_insn        = bytes_per_insn;
    return p;
}

BenchmarkResult run_one(const char* name, JitPool::TestFn fn, const BenchmarkParams& p) {
    if (!fn) {
        printf("%-48s: skipped (JIT compile failed)\n", name);
        fflush(stdout);
        return BenchmarkResult{};
    }
    const BenchmarkResult r = benchmark(fn, name, p);
    g_jit_pool->release(fn);
    return r;
}

void section(const char* title) {
    // Width in terminal columns. The box-drawing dash is 3 bytes but one
    // column, so count code points rather than bytes for the title.
    static constexpr int kWidth = 64;
    int cols = 0;
    for (const unsigned char* s = reinterpret_cast<const unsigned char*>(title); *s; ++s)
        if ((*s & 0xC0) != 0x80) ++cols;   // count UTF-8 lead bytes only

    int dashes = kWidth - 4 - cols;        // "── " + title + " "
    if (dashes < 2) dashes = 2;

    printf("\n── %s ", title);
    for (int i = 0; i < dashes; ++i) fputs("─", stdout);
    printf("\n");
    fflush(stdout);
}

void skip_feature(CpuFeature f, const char* what) {
    printf("  (%s not available on this CPU — skipping %s)\n",
           cpu_feature_name(f), what);
    fflush(stdout);
}

// ── Throughput chain sweeps ───────────────────────────────────────────────────

uint32_t chain_unroll(uint32_t unroll, uint32_t nc, uint32_t group) {
    const uint32_t q = nc * (group ? group : 1u);
    const uint32_t r = (unroll / q) * q;
    return r ? r : q;
}

// ── Page-granular memory helpers ──────────────────────────────────────────────

void* alloc_pages(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

void free_pages(void* p, size_t bytes) {
    if (!p) return;
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

void commit_pages(void* p, size_t bytes) {
    // One byte per 4 KB page. A memset would also warm the caches with a
    // sequential sweep, which the first latency test must not inherit.
    uint8_t* b = static_cast<uint8_t*>(p);
    for (size_t off = 0; off < bytes; off += 4096)
        b[off] = 0;
}

uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

void* build_pointer_ring(void* buf, size_t size, size_t stride, size_t offset) {
    const size_t n_nodes = size / stride;
    if (n_nodes < 2) return nullptr;

    // Index array: 128 MB at 256 B stride is 512K nodes = 2 MB. Fine.
    uint32_t* perm = static_cast<uint32_t*>(malloc(n_nodes * sizeof(uint32_t)));
    if (!perm) {
        fprintf(stderr, "build_pointer_ring: malloc(%zu) failed\n",
                n_nodes * sizeof(uint32_t));
        return nullptr;
    }
    for (uint32_t i = 0; i < n_nodes; ++i) perm[i] = i;

    // Fisher-Yates with the fixed seed: identical chain on every run.
    uint64_t rng = kChaseSeed;
    for (size_t i = n_nodes - 1; i > 0; --i) {
        const size_t j = xorshift64(rng) % (i + 1);
        const uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }

    // Link slot perm[i] → slot perm[i+1]. memcpy because (node + offset)
    // is deliberately unaligned in the misalignment tests, and a direct
    // pointer store there is undefined behaviour even where the hardware
    // is happy with it.
    uint8_t* const base = static_cast<uint8_t*>(buf);
    for (size_t i = 0; i < n_nodes; ++i) {
        uint8_t* slot = base + static_cast<size_t>(perm[i]) * stride + offset;
        const uintptr_t next = reinterpret_cast<uintptr_t>(
            base + static_cast<size_t>(perm[(i + 1) % n_nodes]) * stride + offset);
        memcpy(slot, &next, sizeof(next));
    }

    void* head = base + static_cast<size_t>(perm[0]) * stride + offset;
    free(perm);
    return head;
}

} // namespace arm64bench::gen
