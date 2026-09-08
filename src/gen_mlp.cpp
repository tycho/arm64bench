// gen_mlp.cpp
// Memory-level parallelism via interleaved pointer chases. See gen_mlp.h.
//
// ── Interleaved rings ────────────────────────────────────────────────────────
//
// One random permutation of the buffer's nodes is split round-robin into K
// disjoint cycles: chain j visits perm[j], perm[j+K], perm[j+2K], ... and
// wraps. The union of the chains is the whole buffer whatever K is, so the
// cache level that services the misses depends only on the buffer size, and
// no chain ever revisits a node another chain touched recently (they are
// disjoint). Every timed call walks every node once, for the same reason as
// gen_ooo: a partial walk leaves the tail resident for the next call.
//
// ── Registers ────────────────────────────────────────────────────────────────
//
// Each chain needs a pointer register. x0–x15 are free, x20–x22 are saved by
// build_loop, and x23–x29 are parked in the scratch area by this file's own
// setup/teardown, for 26 chains. That is enough to pass the knee on every
// core measured so far; a "no knee" result means the capacity is ≥ 27.

#include "gen_mlp.h"
#include "gen_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Configuration ─────────────────────────────────────────────────────────────

static constexpr size_t kStride = 256;   // one node per 4 cache lines

struct Level {
    const char* label;
    size_t      bytes;
    bool        resident;   // fits in L1: a load-throughput control, no misses
};
// At a 256-byte stride only one line in four is touched, and only one L1
// set in four is used, so the line footprint is size/4 and the effective L1
// capacity is a quarter of nominal (32 KB on a 128 KB, 8-way L1D). Sizes are
// chosen from this project's latency sweep on Apple M5 with that in mind:
// 64 KB stays L1-resident (a control: load-port throughput, no misses),
// 2 MB misses L1 and hits L2 (~30 clk), 16 MB is the far end of L2 (~50 clk),
// 128 MB is DRAM. Other cores land the same sizes on different levels; the
// label says what the size is and the number says what it cost.
static const Level kLevels[] = {
    { "64KB",   64ULL << 10, true  },
    { "2MB",     2ULL << 20, false },
    { "16MB",   16ULL << 20, false },
    { "128MB", 128ULL << 20, false },
};
static constexpr size_t kMaxBytes = 128ULL << 20;

// Every call walks every chain at least once (see gen_ooo for why), and for
// the cache-resident levels keeps going until it has issued this many loads,
// so a call lasts milliseconds rather than microseconds.
static constexpr uint64_t kMinLoadsPerCall = 16'000'000;

static const Gp kChainRegs[] = {
    x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9,  x10, x11, x12, x13, x14, x15,
    x20, x21, x22, x23, x24, x25, x26, x27, x28, x29,
};
static constexpr uint32_t kMaxChains = sizeof(kChainRegs) / sizeof(kChainRegs[0]);
static constexpr uint32_t kSaveBytes = 64;   // x23–x29 (7 regs, rounded up)

static constexpr uint32_t kChainSweep[] = {
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
};

// ── Ring construction ─────────────────────────────────────────────────────────

// Builds K disjoint random cycles over buf[0..size) at `stride`; writes the
// K chain heads to heads[]. Returns the number of nodes per chain (the loop
// count that walks every chain exactly once), or 0 on failure.
static uint64_t build_interleaved_rings(void* buf, size_t size, size_t stride,
                                        uint32_t k, uintptr_t* heads) {
    const size_t n_nodes = size / stride;
    if (n_nodes < 2 * k) return 0;
    const size_t per_chain = n_nodes / k;       // trailing remainder unused
    const size_t used      = per_chain * k;

    uint32_t* perm = static_cast<uint32_t*>(malloc(used * sizeof(uint32_t)));
    if (!perm) return 0;
    for (uint32_t i = 0; i < used; ++i) perm[i] = i;
    uint64_t rng = kChaseSeed;
    for (size_t i = used - 1; i > 0; --i) {
        const size_t j = xorshift64(rng) % (i + 1);
        const uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }

    uint8_t* const base = static_cast<uint8_t*>(buf);
    for (uint32_t c = 0; c < k; ++c) {
        for (size_t s = 0; s < per_chain; ++s) {
            uint8_t* slot = base + static_cast<size_t>(perm[c + s * k]) * stride;
            const size_t next_idx = (s + 1 < per_chain) ? (c + (s + 1) * k) : c;
            const uintptr_t next = reinterpret_cast<uintptr_t>(
                base + static_cast<size_t>(perm[next_idx]) * stride);
            memcpy(slot, &next, sizeof(next));
        }
        heads[c] = reinterpret_cast<uintptr_t>(base + static_cast<size_t>(perm[c]) * stride);
    }
    free(perm);
    return per_chain;
}

// ── JIT builder ───────────────────────────────────────────────────────────────

static JitPool::TestFn build_chase_k(uint64_t loops, uint32_t k, const uintptr_t* heads) {
    return build_loop_with_teardown(loops, 1,
        [=](a64::Assembler& a) {
            a.stp(x23, x24, ptr(sp, 0));
            a.stp(x25, x26, ptr(sp, 16));
            a.stp(x27, x28, ptr(sp, 32));
            a.str(x29,      ptr(sp, 48));
            for (uint32_t c = 0; c < k; ++c)
                a.mov(kChainRegs[c], Imm(static_cast<uint64_t>(heads[c])));
        },
        [=](a64::Assembler& a, uint32_t) {
            for (uint32_t c = 0; c < k; ++c)
                a.ldr(kChainRegs[c], ptr(kChainRegs[c]));
        },
        [](a64::Assembler& a) {
            a.ldp(x23, x24, ptr(sp, 0));
            a.ldp(x25, x26, ptr(sp, 16));
            a.ldp(x27, x28, ptr(sp, 32));
            a.ldr(x29,      ptr(sp, 48));
        },
        kSaveBytes);
}

// ── Sweep ─────────────────────────────────────────────────────────────────────

static void run_level(const BenchmarkParams& base, void* buf, const Level& lvl) {
    char title[96];
    snprintf(title, sizeof(title), "Memory-level parallelism, %s footprint", lvl.label);
    section(title);

    char name[96];
    static constexpr uint32_t kN = sizeof(kChainSweep) / sizeof(kChainSweep[0]);
    double   t_iter[kN]  = {};    // ns per iteration (K loads); 0 = not run
    double   t_load[kN]  = {};    // ns per load

    for (uint32_t i = 0; i < kN; ++i) {
        const uint32_t k = kChainSweep[i];
        uintptr_t heads[kMaxChains];
        const uint64_t per_chain = build_interleaved_rings(buf, lvl.bytes, kStride, k, heads);
        if (!per_chain) { printf("  (%u chains: buffer too small)\n", k); continue; }

        uint64_t loops = per_chain;
        if (loops * k < kMinLoadsPerCall) loops = (kMinLoadsPerCall + k - 1) / k;
        loops = scale_loops(loops);

        auto fn = build_chase_k(loops, k, heads);
        snprintf(name, sizeof(name), "chase %s x%2u chains", lvl.label, k);
        const BenchmarkResult r = run_one(name, fn, params_for(base, loops, k, 64));
        if (r.min_ns_per_insn <= 0.0) continue;
        t_load[i] = r.min_ns_per_insn;
        t_iter[i] = r.min_ns_per_insn * k;
    }

    // While misses overlap perfectly the iteration costs the same whatever K
    // is; the flat region's floor is the baseline, and the knee is the first
    // K past that floor whose iteration is 25% dearer.
    uint32_t i_min = kN; 
    for (uint32_t i = 0; i < kN; ++i)
        if (t_iter[i] > 0.0 && (i_min == kN || t_iter[i] < t_iter[i_min])) i_min = i;
    if (i_min == kN) return;

    uint32_t i_knee = kN;
    for (uint32_t i = i_min + 1; i < kN; ++i)
        if (t_iter[i] > 0.0 && t_iter[i] > 1.25 * t_iter[i_min]) { i_knee = i; break; }

    uint32_t i_last = kN;
    for (uint32_t i = kN; i-- > 0;) if (t_iter[i] > 0.0) { i_last = i; break; }

    // Best per-load time over the sweep (the floor), not just the last point.
    uint32_t i_best = i_last;
    for (uint32_t i = 0; i < kN; ++i)
        if (t_load[i] > 0.0 && t_load[i] < t_load[i_best]) i_best = i;

    if (lvl.resident) {
        // No misses here: this is how fast dependent loads can issue at all.
        printf("  L1-resident control: 1 chain %.2f ns/load (latency); floor %.3f ns/load"
               " (%.2f loads/ns) from %u chains\n",
               t_load[0], t_load[i_best], 1.0 / t_load[i_best], kChainSweep[i_best]);
        return;
    }

    // Effective parallelism: how many single-chain latencies fit in the time
    // the level needs per load once it is saturated.
    printf("  1 chain: %.1f ns/load; floor %.2f ns/load = %.1f GB/s of lines"
           " → effective MLP ≈ %.1f misses in flight\n",
           t_load[0], t_load[i_best], 64.0 / t_load[i_best], t_load[0] / t_load[i_best]);
    if (i_knee != kN)
        printf("  ↑ iteration time leaves its floor between %u and %u chains\n",
               kChainSweep[i_knee - 1], kChainSweep[i_knee]);
    else
        printf("  (iteration time never leaves its floor up to %u chains)\n",
               kChainSweep[i_last]);
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_mlp_tests(const BenchmarkParams& base_params) {
    void* buf = alloc_pages(kMaxBytes);
    if (!buf) {
        fprintf(stderr, "run_mlp_tests: failed to allocate %zu MB\n", kMaxBytes >> 20);
        return;
    }
    commit_pages(buf, kMaxBytes);

    printf("  ns/insn = ns per load with K independent chases in flight; GB/s = the\n"
           "  cache-line traffic that implies. Per-load time falls as 1/K while the\n"
           "  misses overlap and floors where the level saturates.\n");
    for (const Level& lvl : kLevels)
        run_level(base_params, buf, lvl);

    free_pages(buf, kMaxBytes);
}

} // namespace arm64bench::gen
