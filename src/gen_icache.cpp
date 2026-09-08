// gen_icache.cpp
// Instruction cache and iTLB sweeps. See gen_icache.h.

#include "gen_icache.h"
#include "gen_common.h"
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Code-size sweep ───────────────────────────────────────────────────────────

static constexpr size_t kCodeSizes[] = {
    16ULL << 10, 32ULL << 10, 64ULL << 10, 96ULL << 10, 128ULL << 10, 192ULL << 10,
    256ULL << 10, 384ULL << 10, 512ULL << 10, 768ULL << 10,
    1ULL << 20, 2ULL << 20, 4ULL << 20, 8ULL << 20, 16ULL << 20,
};
static constexpr uint64_t kInsnsPerCall = 64'000'000;   // ~8 ms at 10/clk, 4 GHz

static void run_code_size_sweep(const BenchmarkParams& base) {
    section("I-cache: straight-line NOP body, clk per instruction vs code size");
    printf("  Each body is executed as one loop iteration; the loop repeats it so the\n"
           "  footprint is exactly the body. Front-end width bounds the flat region.\n");

    char name[80];
    for (const size_t bytes : kCodeSizes) {
        const uint32_t insns = static_cast<uint32_t>(bytes / 4);
        uint64_t loops = kInsnsPerCall / insns;
        if (loops < 2) loops = 2;
        loops = scale_loops(loops);

        auto fn = build_loop(loops, insns, no_setup,
                             [](a64::Assembler& a, uint32_t) { a.nop(); });
        if (bytes >= (1ULL << 20))
            snprintf(name, sizeof(name), "NOP body %4zu MB", bytes >> 20);
        else
            snprintf(name, sizeof(name), "NOP body %4zu KB", bytes >> 10);
        run_one(name, fn, params_for(base, loops, insns));
    }
}

// ── Page-chain sweep ──────────────────────────────────────────────────────────
//
// Layout of the loop body for N pages:
//
//   page 0:  <pad>  L0: b L1   ; <pad to 16 KB>
//   page 1:  <pad>  L1: b L2   ; <pad>
//   ...
//   page N-1: <pad> LN-1: b LN ; <pad>
//   LN:      (loop control follows)
//
// The pad is zero words (UDF #0) embedded in bulk, never executed. 16 KB
// spacing is one page on macOS and four on 4 KB-page Linux/Windows, so each
// branch always lands on a fresh page for the TLB either way. Within its
// page each branch sits at a different line offset (p × 5 lines, wrapping),
// because a 16 KB stride from a fixed offset maps every branch onto the same
// two I-cache sets and the sweep would measure set conflicts instead of TLBs.

static constexpr size_t   kPageStride = 16ULL << 10;
static constexpr uint32_t kPageCounts[] = {
    4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
};

// Two strides, same chain lengths. Dense (64 B, one line per branch) has
// N branch sites but only N lines and a handful of pages, so its cost is
// the branch predictor's target-buffer capacity alone. Paged (16 KB) has
// the same N sites plus N instruction pages; paged minus dense is the TLB.
static void run_chain_sweep(const BenchmarkParams& base, size_t stride, const char* label) {
    static uint8_t zeros[kPageStride];   // UDF #0 padding, never executed
    memset(zeros, 0, sizeof(zeros));

    char name[80];
    for (const uint32_t pages : kPageCounts) {
        // ~2 M branches per call: 2M × (2–50 clk) = 1–25 ms.
        uint64_t loops = 2'000'000 / pages;
        if (loops < 2) loops = 2;
        loops = scale_loops(loops);

        auto fn = build_loop(loops, 1, no_setup,
            [pages, stride](a64::Assembler& a, uint32_t) {
                // labels[p] marks branch p; labels[pages] is the exit.
                Label labels[2049];
                for (uint32_t p = 0; p <= pages; ++p) labels[p] = a.new_label();
                for (uint32_t p = 0; p < pages; ++p) {
                    const size_t off = (static_cast<size_t>(p) * 5 * 64) % stride;
                    if (off) a.embed(zeros, off);
                    a.bind(labels[p]);
                    a.b(labels[p + 1]);
                    a.embed(zeros, stride - off - 4);
                }
                a.bind(labels[pages]);
            });
        const size_t span = pages * stride;
        if (span >= (1ULL << 20))
            snprintf(name, sizeof(name), "B chain %s %5u sites (%3zu MB)", label, pages, span >> 20);
        else
            snprintf(name, sizeof(name), "B chain %s %5u sites (%3zu KB)", label, pages, span >> 10);
        run_one(name, fn, params_for(base, loops, pages));
    }
}

static void run_page_chain_sweep(const BenchmarkParams& base) {
    section("BTB: taken-branch chain, 64 B apart (target-buffer capacity)");
    printf("  N distinct taken branches on N adjacent lines; clk per branch rises\n"
           "  where N outgrows the branch target buffers.\n");
    run_chain_sweep(base, 64, "dense");

    section("iTLB: the same chain, one 16 KB page per branch");
    printf("  Same N sites, now on N pages. Cost above the dense chain at the same N\n"
           "  is instruction-TLB reach (L1 iTLB, then L2 TLB, then page walks).\n");
    run_chain_sweep(base, kPageStride, "paged");
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_icache_tests(const BenchmarkParams& base_params) {
    run_code_size_sweep(base_params);
    run_page_chain_sweep(base_params);
}

} // namespace arm64bench::gen
