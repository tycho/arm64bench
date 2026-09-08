// gen_bf16.cpp
// FEAT_BF16 microbenchmarks. See gen_bf16.h.
//
// Operand setup: bf16 has the same exponent layout as f32 with the low 16
// mantissa bits dropped, so MOVI #0x3F, LSL #8 (0x3F00) is bf16 0.5 in every
// lane. Products of 0.5 × 0.5 accumulate slowly and stay finite for the
// length of any test here.
//
// MAC accounting per instruction (v4s destination, v8h inputs):
//   BFDOT    4 lanes × 2 products  =  8 MACs
//   BFMMLA   2×2 result × 4 products = 16 MACs
//   BFMLALB  4 lanes × 1 product   =  4 MACs

#include "gen_bf16.h"
#include "gen_common.h"
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

static void seed_bf16(a64::Assembler& a, uint32_t nc) {
    // Inputs in vr(nc), vr(nc+1); accumulators vr(0..nc-1) zeroed.
    a.movi(vr(nc    ).h8(), Imm(0x3F), Imm(8));
    a.movi(vr(nc + 1).h8(), Imm(0x3F), Imm(8));
    for (uint32_t i = 0; i < nc; ++i) a.movi(vr(i).s4(), Imm(0));
}

static void run_bf16_section(const BenchmarkParams& base,
                             uint64_t loops, uint32_t unroll) {
    if (!cpu_has(CpuFeature::BF16)) {
        skip_feature(CpuFeature::BF16, "BFDOT/BFMMLA/BFMLALB/BFMLALT");
        return;
    }

    char name[80];

    // ── BFDOT v4s latency ─────────────────────────────────────────────────
    // BFDOT Vd.4S, Vn.8H, Vm.8H — Vd accumulates (chains).
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { seed_bf16(a, 1); },
            [](a64::Assembler& a, uint32_t) { a.bfdot(vr(0).s4(), vr(1).h8(), vr(2).h8()); });
        snprintf(name, sizeof(name), "BFDOT v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    chain_sweep(base, loops, unroll, "BFDOT v4s tput", { 2, 3, 4, 6 },
        [](a64::Assembler& a, uint32_t nc) { seed_bf16(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.bfdot(vr(u % nc).s4(), vr(nc).h8(), vr(nc + 1).h8()); });

    // ── BFMMLA v4s latency ────────────────────────────────────────────────
    // BFMMLA Vd.4S, Vn.8H, Vm.8H — 2×4 × 4×2 bf16 matrices, 2×2 f32 result
    // accumulated into Vd (chains). Twice the MACs of BFDOT per instruction.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { seed_bf16(a, 1); },
            [](a64::Assembler& a, uint32_t) { a.bfmmla(vr(0).s4(), vr(1).h8(), vr(2).h8()); });
        snprintf(name, sizeof(name), "BFMMLA v4s latency    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    chain_sweep(base, loops, unroll, "BFMMLA v4s tput", { 2, 3, 4, 6 },
        [](a64::Assembler& a, uint32_t nc) { seed_bf16(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.bfmmla(vr(u % nc).s4(), vr(nc).h8(), vr(nc + 1).h8()); });

    // ── BFMLALB / BFMLALT v4s latency ─────────────────────────────────────
    // Widening multiply-accumulate of the even (B) or odd (T) bf16 lanes
    // into f32; Vd accumulates (chains). The pair is how a full bf16 vector
    // is consumed without BFDOT's pairwise sum.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { seed_bf16(a, 1); },
            [](a64::Assembler& a, uint32_t) { a.bfmlalb(vr(0).s4(), vr(1).h8(), vr(2).h8()); });
        snprintf(name, sizeof(name), "BFMLALB v4s latency   (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { seed_bf16(a, 1); },
            [](a64::Assembler& a, uint32_t) { a.bfmlalt(vr(0).s4(), vr(1).h8(), vr(2).h8()); });
        snprintf(name, sizeof(name), "BFMLALT v4s latency   (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    chain_sweep(base, loops, unroll, "BFMLALB v4s tput", { 2, 3, 4, 6 },
        [](a64::Assembler& a, uint32_t nc) { seed_bf16(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.bfmlalb(vr(u % nc).s4(), vr(nc).h8(), vr(nc + 1).h8()); });
}

// ── Public entry point ────────────────────────────────────────────────────────

void run_bf16_tests(const BenchmarkParams& base_params) {
    run_bf16_section(base_params, base_params.loops,
                     base_params.instructions_per_loop);
}

} // namespace arm64bench::gen
