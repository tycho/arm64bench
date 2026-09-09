// gen_ooo.cpp
// Out-of-order window sizing via the two-miss probe. See gen_ooo.h.
//
// ── Why two chains, not one ──────────────────────────────────────────────────
//
// A single dependent pointer chain serialises on its own loads regardless of
// window size. What the window buys is overlap between *independent* misses:
// chain A's load A_i and chain B's load B_i. B_i can only issue while A_i is
// outstanding if the front-end has managed to bring B_i into the window,
// i.e. if the N fillers between them (plus the two loads) fit. A_{i+1}
// depends on A_i and cannot overlap anyway, so the knee is at N ≈ R − 2.
//
// ── Keeping the fillers cheap ────────────────────────────────────────────────
//
// The fillers must execute entirely in the shadow of the miss, or their own
// throughput cost would masquerade as the knee. Each chain therefore takes
// kMissesPerChain back-to-back dependent misses per iteration (~750 cycles of
// shadow at 4.4 GHz for two), against ≤ 2 × 2048 NOPs at 8-wide decode or
// 2 × 1024 ADDs at 6/cycle. Beyond a knee the curve still drifts up with N
// for this reason — the estimate is taken from the first jump, not the plateau.
//
// ── Rings ────────────────────────────────────────────────────────────────────
//
// Two disjoint random pointer rings, one per chain, each spanning 64 MB at a
// 256-byte stride. 64 MB exceeds every cache on the target cores and, at
// 16 KB pages, exceeds the L2 TLB, so each load is a DRAM miss plus a page
// walk — the longest shadow available. The rings used for the sweeps store
// XOR-masked links (mask_pointer_ring) so a data-dependent prefetcher cannot
// shorten the misses; a plain ring is kept only for the DMP check.

#include "gen_ooo.h"
#include "gen_common.h"
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Configuration ─────────────────────────────────────────────────────────────

static constexpr size_t   kRingBytes   = 64ULL << 20;      // per chain
static constexpr size_t   kBufBytes    = 3 * kRingBytes;   // one plain ring + two masked rings
static constexpr size_t   kRingStride  = 256;
static constexpr uint32_t kScratch     = 64;               // L1-hot slot for LDR/STR fillers
static constexpr uint64_t kRingNodes   = kRingBytes / kRingStride;

// Dependent misses per chain per iteration. Two back-to-back misses double
// the shadow the fillers must hide in (~750 cycles at 4.4 GHz), which keeps
// 2 × 1024 ADD fillers (≈340 cycles at 6/cycle) comfortably inside it.
static constexpr uint32_t kMissesPerChain = 2;

// Every timed call must walk the WHOLE ring: a partial walk revisits the
// same nodes every call, the per-sample warm-up leaves them resident in L2,
// and a "DRAM miss" quietly becomes an L2 hit (27 ns instead of 81 ns on
// M5). One full traversal per call at ~85 ns/miss is ~22 ms.
static constexpr uint64_t kLoopsPerCall = kRingNodes / kMissesPerChain;

// Filler GPRs: x2–x8, x10–x15. x0/x1 are the chains, x9 the scratch pointer.
static const Gp kFillGp[] = { x2, x3, x4, x5, x6, x7, x8, x10, x11, x12, x13, x14, x15 };
static constexpr uint32_t kNumFillGp = sizeof(kFillGp) / sizeof(kFillGp[0]);

// Filler FP registers: caller-saved, v8–v15 avoided. d22 is the constant addend.
static const Vec kFillFp[] = { d0, d1, d2, d3, d4, d5, d6, d7,
                               d16, d17, d18, d19, d20, d21 };
static constexpr uint32_t kNumFillFp = sizeof(kFillFp) / sizeof(kFillFp[0]);

enum class Filler { Nop, IntAdd, FpAdd, Load, Store, Cmp, Bcond };

struct FillerKind {
    Filler       filler;
    const char*  label;      // test-name prefix
    const char*  structure;  // what the knee measures
    std::span<const uint32_t> sweep;
};

// Coarse sweep for ROB-class structures (hundreds to a couple of thousand) …
static constexpr uint32_t kWindowSweep[] = {
     16,   32,   48,   64,   96,  128,  160,  192,  224,  256,  288,  320,
    352,  384,  416,  448,  480,  512,  576,  640,  704,  768,  832,  896,
    960, 1024, 1152, 1280, 1408, 1536, 1792, 2048,
};
// … and a finer one for the load/store queues (tens to a few hundred).
static constexpr uint32_t kQueueSweep[] = {
      8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96, 104, 112, 120, 128,
    136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
    288, 320, 352, 384, 416, 448, 480, 512,
};

static const FillerKind kKinds[] = {
    { Filler::Nop,    "ROB   (NOP fill)  ", "reorder buffer",          kWindowSweep },
    { Filler::IntAdd, "IntPRF (ADD fill) ", "integer register file",   kWindowSweep },
    { Filler::FpAdd,  "FpPRF (FADD fill) ", "FP/SIMD register file",   kWindowSweep },
    { Filler::Load,   "LoadQ (LDR fill)  ", "load queue",              kQueueSweep  },
    { Filler::Store,  "StoreQ (STR fill) ", "store queue",             kQueueSweep  },
    { Filler::Cmp,    "FlagPRF (CMP fill)", "flag register file",      kQueueSweep  },
    { Filler::Bcond,  "BOB   (B.cond fill)", "branch order buffer",    kQueueSweep  },
};

// ── JIT builders ──────────────────────────────────────────────────────────────

static void emit_filler(a64::Assembler& a, Filler f, uint32_t k) {
    switch (f) {
        case Filler::Nop:    a.nop(); break;
        case Filler::IntAdd: { const Gp& r = kFillGp[k % kNumFillGp]; a.add(r, r, Imm(1)); break; }
        case Filler::FpAdd:  { const Vec& d = kFillFp[k % kNumFillFp]; a.fadd(d, d, d22); break; }
        // LDR to XZR: a real load-queue entry with no physical register to
        // allocate, so the knee cannot be the integer PRF in disguise.
        case Filler::Load:   a.ldr(xzr, ptr(x9)); break;
        case Filler::Store:  a.str(x2, ptr(x9, static_cast<int32_t>(8 * (k % 8)))); break;
        // CMP writes NZCV only: a flag physical register and a ROB entry.
        case Filler::Cmp:    a.cmp(x2, x3); break;
        // B.EQ with flags NE (set once in setup, x2 = 1, x3 = 2, and the CMP
        // filler reproduces them): never taken, a branch-order-buffer entry
        // and no register.
        case Filler::Bcond: {
            Label l = a.new_label();
            a.b(CondCode::kEQ, l);
            a.bind(l);
            break;
        }
    }
}

static void emit_setup(a64::Assembler& a, uintptr_t head_a, uintptr_t head_b) {
    a.mov(x0, Imm(static_cast<uint64_t>(head_a)));
    a.mov(x1, Imm(static_cast<uint64_t>(head_b)));
    a.mov(x20, Imm(kPtrMask));
    a.mov(x9, sp);                                   // scratch slot (L1-hot)
    for (uint32_t i = 0; i < kNumFillGp; ++i) a.mov(kFillGp[i], Imm(i + 1));
    a.fmov(d22, 1.0);
    for (uint32_t i = 0; i < kNumFillFp; ++i) a.fmov(kFillFp[i], 1.0);
    a.str(x2, ptr(x9));                              // make the slot resident
    a.cmp(x2, x3);                                   // NE, and nothing in the loop but the CMP filler writes flags
}

// One iteration = miss A, N fillers, miss B, N fillers.
static JitPool::TestFn build_pair_probe(uint64_t loops, uintptr_t head_a, uintptr_t head_b,
                                        Filler f, uint32_t n_fill, bool obf) {
    return build_loop(loops, 1,
        [=](a64::Assembler& a) { emit_setup(a, head_a, head_b); },
        [=](a64::Assembler& a, uint32_t) {
            for (uint32_t m = 0; m < kMissesPerChain; ++m) {
                a.ldr(x0, ptr(x0));
                if (obf) a.eor(x0, x0, x20);
            }
            for (uint32_t k = 0; k < n_fill; ++k) emit_filler(a, f, k);
            for (uint32_t m = 0; m < kMissesPerChain; ++m) {
                a.ldr(x1, ptr(x1));
                if (obf) a.eor(x1, x1, x20);
            }
            for (uint32_t k = 0; k < n_fill; ++k) emit_filler(a, f, n_fill + k);
        },
        kScratch);
}

// Reference: one chain alone (the cost of a single miss). Run on both a
// plain ring and a masked ring: a data-dependent prefetcher (Apple M-series
// "DMP", the GoFetch mechanism) can only follow pointer-looking values, so
// a plain chase that is markedly faster than a masked one exposes it — and
// is why every sweep below runs on masked rings.
static JitPool::TestFn build_single_chase(uint64_t loops, uintptr_t head, bool obf) {
    return build_loop(loops, 1,
        [=](a64::Assembler& a) {
            a.mov(x0, Imm(static_cast<uint64_t>(head)));
            a.mov(x20, Imm(kPtrMask));
        },
        [=](a64::Assembler& a, uint32_t) {
            for (uint32_t m = 0; m < kMissesPerChain; ++m) {
                a.ldr(x0, ptr(x0));
                if (obf) a.eor(x0, x0, x20);
            }
        });
}

// ── Sweep ─────────────────────────────────────────────────────────────────────

static void run_window_sweep(const BenchmarkParams& base, uint64_t loops,
                             uintptr_t head_a, uintptr_t head_b,
                             const FillerKind& kind, double pair_ns) {
    char name[96];

    // Knee: first N whose time exceeds the overlapped pair by half the gap
    // between overlapped (1×) and serialised (2×). Without a reference
    // (e.g. it was filtered out) there is nothing to compare against.
    const bool   detect    = pair_ns > 0.0;
    const double threshold = pair_ns * 1.5;
    uint32_t prev_n = 0;
    bool     found  = !detect;

    for (const uint32_t n : kind.sweep) {
        auto fn = build_pair_probe(loops, head_a, head_b, kind.filler, n, true);
        snprintf(name, sizeof(name), "%s N=%4u", kind.label, n);
        const BenchmarkResult r = run_one(name, fn, params_for(base, loops, kMissesPerChain));

        if (!found && r.min_ns_per_insn > 0.0 && r.min_ns_per_insn > threshold) {
            // Chain B's first load sits N + kMissesPerChain instructions after
            // chain A's first load, so it overlaps A while the window holds
            // N + kMissesPerChain + 1; the jump happened somewhere in (prev_n, n].
            printf("  ↑ %s exhausted between N=%u and N=%u → capacity ≈ %u–%u entries\n",
                   kind.structure, prev_n, n,
                   prev_n + kMissesPerChain, n + kMissesPerChain + 1);
            found = true;
        }
        prev_n = n;
    }
    if (detect && !found)
        printf("  (no knee up to N=%u — %s is larger than the sweep, or the fillers\n"
               "   are not the limiting structure on this core)\n",
               prev_n, kind.structure);
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_ooo_tests(const BenchmarkParams& base_params) {
    void* buf = alloc_pages(kBufBytes);
    if (!buf) {
        fprintf(stderr, "run_ooo_tests: failed to allocate %zu MB\n", kBufBytes >> 20);
        return;
    }
    commit_pages(buf, kBufBytes);

    uint8_t* const b = static_cast<uint8_t*>(buf);
    void* head_plain = build_pointer_ring(b,                 kRingBytes, kRingStride);
    void* head_a     = build_pointer_ring(b + 1 * kRingBytes, kRingBytes, kRingStride);
    void* head_b     = build_pointer_ring(b + 2 * kRingBytes, kRingBytes, kRingStride);
    if (!head_plain || !head_a || !head_b) {
        fprintf(stderr, "run_ooo_tests: failed to build pointer rings\n");
        free_pages(buf, kBufBytes);
        return;
    }
    mask_pointer_ring(head_a);
    mask_pointer_ring(head_b);
    const uintptr_t hp = reinterpret_cast<uintptr_t>(head_plain);
    const uintptr_t ha = reinterpret_cast<uintptr_t>(head_a);
    const uintptr_t hb = reinterpret_cast<uintptr_t>(head_b);

    const uint64_t loops = scale_loops(kLoopsPerCall);

    section("Out-of-order window (two-miss probe)");
    printf("  ns/insn = one DRAM miss (single chain) or one miss PAIR (two chains).\n"
           "  A pair costs ~1 miss while N+%u fits the window and ~2 once it does not.\n",
           kMissesPerChain);

    const BenchmarkResult plain =
        run_one("single chain, plain pointers (DMP check)", build_single_chase(loops, hp, false),
                params_for(base_params, loops, kMissesPerChain));
    const BenchmarkResult single =
        run_one("single chain, masked pointers (reference)", build_single_chase(loops, ha, true),
                params_for(base_params, loops, kMissesPerChain));
    const BenchmarkResult pair =
        run_one("two chains N=0, masked pointers (reference)",
                build_pair_probe(loops, ha, hb, Filler::Nop, 0, true),
                params_for(base_params, loops, kMissesPerChain));

    if (plain.min_ns_per_insn > 0.0 && single.min_ns_per_insn > 0.0 &&
        plain.min_ns_per_insn < 0.8 * single.min_ns_per_insn)
        printf("  ! plain chase is %.0f%% faster than masked: a data-dependent prefetcher is\n"
               "    following the pointers (sweeps below use masked rings and are unaffected)\n",
               (1.0 - plain.min_ns_per_insn / single.min_ns_per_insn) * 100.0);
    if (single.min_ns_per_insn > 0.0 && pair.min_ns_per_insn > 0.0)
        printf("  overlap factor = %.2f (pair / single; ~1.0 = fully overlapped, the sweeps'\n"
               "  baseline; >1.3 means the method is not working on this core)\n",
               pair.min_ns_per_insn / single.min_ns_per_insn);

    for (const FillerKind& kind : kKinds) {
        char title[96];
        snprintf(title, sizeof(title), "%s — %s", kind.structure, kind.label);
        section(title);
        run_window_sweep(base_params, loops, ha, hb, kind, pair.min_ns_per_insn);
    }

    free_pages(buf, kBufBytes);
}

} // namespace arm64bench::gen
