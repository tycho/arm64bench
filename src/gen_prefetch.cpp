// gen_prefetch.cpp
// Prefetcher characterization. See gen_prefetch.h.
//
// ── Stride streams: the tiling ───────────────────────────────────────────────
//
// For a stride S (a multiple of 64) and run length R, the buffer is cut into
// windows of R × S bytes. Inside a window there are S/64 runs: run k visits
// lines k×64, k×64 + S, k×64 + 2S, ... (R of them). Every line in the window
// belongs to exactly one run, so a pass over all windows and all runs touches
// every line once, and a 64 MB buffer at 256 B stride costs the same number
// of DRAM lines as at 16 KB stride. Windows and runs within a window are
// visited in random order (fixed seed); the only regularity a prefetcher can
// exploit is the R consecutive strided accesses of a run.
//
// The chain is stored in the buffer exactly like every other chase test:
// each visited line holds the address of the next.

#include "gen_prefetch.h"
#include "gen_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Configuration ─────────────────────────────────────────────────────────────

static constexpr size_t   kLine       = 64;
static constexpr size_t   kStreamBuf  = 64ULL << 20;   // 1 M lines: past L2 and SLC
static constexpr uint32_t kRunLen     = 32;
static constexpr size_t   kStrides[]  = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
};

static constexpr size_t   kChaseBuf    = 128ULL << 20;
static constexpr size_t   kChaseStride = 256;
static constexpr uint32_t kLookaheads[] = { 0, 1, 2, 4, 8, 16, 32 };

// ── Stride-stream chain ───────────────────────────────────────────────────────

// Links the tiled runs into one ring; returns the head and the node count.
static void* build_stream_ring(void* buf, size_t size, size_t stride, uint32_t run_len,
                               bool backward, uint64_t* out_nodes) {
    const size_t window   = stride * run_len;
    const size_t windows  = size / window;
    const size_t runs_per = stride / kLine;
    const size_t total    = windows * runs_per * run_len;   // == size / kLine
    if (!windows || !total) return nullptr;

    // Random order of (window, run) pairs.
    const size_t n_runs = windows * runs_per;
    uint32_t* order = static_cast<uint32_t*>(malloc(n_runs * sizeof(uint32_t)));
    if (!order) return nullptr;
    for (uint32_t i = 0; i < n_runs; ++i) order[i] = i;
    uint64_t rng = kChaseSeed;
    for (size_t i = n_runs - 1; i > 0; --i) {
        const size_t j = xorshift64(rng) % (i + 1);
        const uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
    }

    uint8_t* const base = static_cast<uint8_t*>(buf);
    auto node_addr = [&](size_t run_idx, uint32_t step) -> uint8_t* {
        const size_t w = run_idx / runs_per;
        const size_t k = run_idx % runs_per;
        const uint32_t s = backward ? (run_len - 1 - step) : step;
        return base + w * window + k * kLine + static_cast<size_t>(s) * stride;
    };

    uint8_t* head = node_addr(order[0], 0);
    uint8_t* prev = nullptr;
    for (size_t i = 0; i < n_runs; ++i) {
        for (uint32_t step = 0; step < run_len; ++step) {
            uint8_t* cur = node_addr(order[i], step);
            if (prev) { const uintptr_t p = reinterpret_cast<uintptr_t>(cur); memcpy(prev, &p, sizeof(p)); }
            prev = cur;
        }
    }
    { const uintptr_t p = reinterpret_cast<uintptr_t>(head); memcpy(prev, &p, sizeof(p)); }
    free(order);
    *out_nodes = total;
    return head;
}

static void run_stride_streams(const BenchmarkParams& base, void* buf, double random_ns) {
    char name[96];
    for (const bool backward : { false, true }) {
        snprintf(name, sizeof(name), "Stride streams, %s, runs of %u, 64 MB footprint",
                 backward ? "descending" : "ascending", kRunLen);
        section(name);
        if (random_ns > 0.0)
            printf("  Unassisted random chase on this machine: %.1f ns/load. Anything well\n"
                   "  below that is the hardware prefetcher following the stream.\n", random_ns);
        for (const size_t stride : kStrides) {
            uint64_t nodes = 0;
            void* head = build_stream_ring(buf, kStreamBuf, stride, kRunLen, backward, &nodes);
            if (!head) { printf("  (stride %zu: could not tile)\n", stride); continue; }

            const uint64_t loops = scale_loops(nodes);   // one full pass per call
            auto fn = build_loop(loops, 1,
                [head](a64::Assembler& a) { a.mov(x0, Imm(reinterpret_cast<uint64_t>(head))); },
                [](a64::Assembler& a, uint32_t) { a.ldr(x0, ptr(x0)); });
            if (stride >= 1024)
                snprintf(name, sizeof(name), "%s stride %3zu KB", backward ? "desc" : "asc ", stride >> 10);
            else
                snprintf(name, sizeof(name), "%s stride %3zu B ", backward ? "desc" : "asc ", stride);
            run_one(name, fn, params_for(base, loops, 1, 64));
        }
    }
}

// ── PRFM lookahead ────────────────────────────────────────────────────────────

// Random ring where node i (in visit order) holds next at +0 and the node
// D hops ahead at +8. D = 0 stores the node itself (a harmless prefetch of
// a line that is about to be loaded anyway).
static void* build_lookahead_ring(void* buf, size_t size, size_t stride, uint32_t d) {
    const size_t n = size / stride;
    uint32_t* perm = static_cast<uint32_t*>(malloc(n * sizeof(uint32_t)));
    if (!perm) return nullptr;
    for (uint32_t i = 0; i < n; ++i) perm[i] = i;
    uint64_t rng = kChaseSeed;
    for (size_t i = n - 1; i > 0; --i) {
        const size_t j = xorshift64(rng) % (i + 1);
        const uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    uint8_t* const base = static_cast<uint8_t*>(buf);
    for (size_t i = 0; i < n; ++i) {
        uint8_t* node = base + static_cast<size_t>(perm[i]) * stride;
        const uintptr_t next  = reinterpret_cast<uintptr_t>(base + static_cast<size_t>(perm[(i + 1) % n]) * stride);
        const uintptr_t ahead = reinterpret_cast<uintptr_t>(base + static_cast<size_t>(perm[(i + d) % n]) * stride);
        memcpy(node,     &next,  sizeof(next));
        memcpy(node + 8, &ahead, sizeof(ahead));
    }
    void* head = base + static_cast<size_t>(perm[0]) * stride;
    free(perm);
    return head;
}

// Returns the unassisted (D = 0) ns per load, for the stride section's reference line.
static double run_prfm_lookahead(const BenchmarkParams& base, void* buf) {
    section("PRFM PLDL1KEEP lookahead on a random DRAM chase, 128 MB");
    printf("  Each node also holds the address D hops ahead; PRFM it, then load next.\n"
           "  D = 0 is the unassisted chase. Per-load time falling with D = honored.\n");

    const uint64_t nodes = kChaseBuf / kChaseStride;
    char name[96];
    double random_ns = 0.0;
    for (const uint32_t d : kLookaheads) {
        void* head = build_lookahead_ring(buf, kChaseBuf, kChaseStride, d);
        if (!head) continue;
        const uint64_t loops = scale_loops(nodes);
        auto fn = build_loop(loops, 1,
            [head](a64::Assembler& a) { a.mov(x0, Imm(reinterpret_cast<uint64_t>(head))); },
            [](a64::Assembler& a, uint32_t) {
                a.ldr(x1, ptr(x0, 8));                                   // address D ahead
                a.prfm(Predicate::PRFOp::kPLDL1KEEP, ptr(x1));
                a.ldr(x0, ptr(x0));                                      // the chase
            });
        snprintf(name, sizeof(name), "PRFM lookahead D=%2u", d);
        const BenchmarkResult r = run_one(name, fn, params_for(base, loops, 1, 64));
        if (d == 0) random_ns = r.min_ns_per_insn;
    }
    return random_ns;
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_prefetch_tests(const BenchmarkParams& base_params) {
    void* buf = alloc_pages(kChaseBuf);
    if (!buf) {
        fprintf(stderr, "run_prefetch_tests: failed to allocate %zu MB\n", kChaseBuf >> 20);
        return;
    }
    commit_pages(buf, kChaseBuf);

    printf("  ns/insn = ns per dependent load; GB/s = the line traffic that implies.\n");
    const double random_ns = run_prfm_lookahead(base_params, buf);
    run_stride_streams(base_params, buf, random_ns);

    free_pages(buf, kChaseBuf);
}

} // namespace arm64bench::gen
