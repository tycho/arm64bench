// gen_i8mm.cpp
// FEAT_I8MM microbenchmarks: USDOT and the SMMLA/UMMLA/USMMLA 2x8x8x2 matrix
// multiply-accumulate family. Split out of gen_fp_simd.cpp; runs as part of
// --simd and skips itself when the CPU lacks FEAT_I8MM.

#include "gen_i8mm.h"
#include "gen_common.h"
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ════════════════════════════════════════════════════════════════════════════
// Section 10: FEAT_I8MM — integer 8-bit matrix multiply (ARMv8.6-A / ARMv9.2)
// ════════════════════════════════════════════════════════════════════════════
//
// FEAT_I8MM adds mixed-signedness dot products and matrix multiply-accumulate.
// Available on M2+ (ARMv8.6-A mandatory), Snapdragon X Elite, Graviton3+.
//
// ── USDOT (unsigned × signed dot product) ────────────────────────────────
//
// USDOT Vd.4S, Vn.16B, Vm.16B — Vd[i] += unsigned(Vn[4i:4i+4]) · signed(Vm[4i:4i+4])
//   The direct ARM equivalent of Intel AVX-VNNI VPDPBUSD: unsigned activations
//   multiplied by signed weights. SDOT/UDOT require equal sign on both operands,
//   forcing a zero-point bias adjustment for asymmetric quantization; USDOT does not.
//
// ── SMMLA / UMMLA / USMMLA (8-bit matrix multiply-accumulate) ─────────────
//
// SMMLA Vd.4S, Vn.16B, Vm.16B — 2×8 signed matrix × 8×2 signed matrix → 2×2 int32
//   Vd.s4() holds a 2×2 int32 result matrix packed as [row0col0, row0col1, row1col0, row1col1].
//   Each element accumulates 8 int8×int8 products — 2× the depth of SDOT.
//   Effective MAC throughput: 32 ops per instruction vs 16 for SDOT (same register width).
//
// UMMLA:  both operands unsigned. USMMLA: Vn unsigned, Vm signed (the ML-critical form).
//
// ── Windows detection ─────────────────────────────────────────────────────
//
// No PF_ARM_I8MM_INSTRUCTIONS_AVAILABLE exists in the Windows SDK. The correct
// proxy is PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE (WinSDK 10.0.26100+): SVE-I8MM
// implies plain I8MM. Used by FFmpeg, dav1d, and others for the same purpose.


static void run_i8mm_section(const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    if (!cpu_has(CpuFeature::I8MM)) {
        skip_feature(CpuFeature::I8MM, "USDOT/SMMLA/UMMLA/USMMLA");
        return;
    }

    char name[80];

    // ── USDOT v4s latency ─────────────────────────────────────────────────
    // USDOT V0.4S, V1.16B, V2.16B — unsigned(V1) · signed(V2) dot product.
    // V1 = constant unsigned bytes, V2 = constant signed bytes. V0 chains.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(2).b16(), Imm(0x03));
                a.movi(vr(1).b16(), Imm(0x02));
                a.movi(vr(0).s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.usdot(vr(0).s4(), vr(1).b16(), vr(2).b16());
            });
        snprintf(name, sizeof(name), "USDOT v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── USDOT v4s throughput: sweep 2..6 chains ───────────────────────────
    {
        chain_sweep(base, loops, unroll, "USDOT v4s tput", { 2, 3, 4, 6, 8, 12, 16 },
            [](a64::Assembler& a, uint32_t nc) {
                a.movi(vr(nc + 1).b16(), Imm(0x03));
                a.movi(vr(nc    ).b16(), Imm(0x02));
                for (uint32_t i = 0; i < nc; ++i)
                    a.movi(vr(i).s4(), Imm(0));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.usdot(vr(u % nc).s4(), vr(nc).b16(), vr(nc + 1).b16());
            });
    }

    // ── SMMLA v4s latency ─────────────────────────────────────────────────
    // SMMLA V0.4S, V1.16B, V2.16B — 2×8 signed × 8×2 signed matrix MLA.
    // Each of the 4 int32 accumulators sums 8 int8×int8 products (vs 4 for SDOT).
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(2).b16(), Imm(0x02));
                a.movi(vr(1).b16(), Imm(0x03));
                a.movi(vr(0).s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.smmla(vr(0).s4(), vr(1).b16(), vr(2).b16());
            });
        snprintf(name, sizeof(name), "SMMLA v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── SMMLA v4s throughput: sweep 2..6 chains ───────────────────────────
    {
        chain_sweep(base, loops, unroll, "SMMLA v4s tput", { 2, 3, 4, 6, 8, 12, 16 },
            [](a64::Assembler& a, uint32_t nc) {
                a.movi(vr(nc + 1).b16(), Imm(0x02));
                a.movi(vr(nc    ).b16(), Imm(0x03));
                for (uint32_t i = 0; i < nc; ++i)
                    a.movi(vr(i).s4(), Imm(0));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.smmla(vr(u % nc).s4(), vr(nc).b16(), vr(nc + 1).b16());
            });
    }

    // ── UMMLA v4s latency ─────────────────────────────────────────────────
    // UMMLA V0.4S, V1.16B, V2.16B — unsigned × unsigned matrix MLA.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(2).b16(), Imm(0x02));
                a.movi(vr(1).b16(), Imm(0x03));
                a.movi(vr(0).s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.ummla(vr(0).s4(), vr(1).b16(), vr(2).b16());
            });
        snprintf(name, sizeof(name), "UMMLA v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── USMMLA v4s latency ────────────────────────────────────────────────
    // USMMLA V0.4S, V1.16B, V2.16B — unsigned(V1) × signed(V2) matrix MLA.
    // Matrix-multiply form of USDOT: 2× MAC depth per instruction.
    // The key instruction for INT8 quantized GEMM with asymmetric quantization.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(2).b16(), Imm(0x03));   // signed weights
                a.movi(vr(1).b16(), Imm(0x02));   // unsigned activations
                a.movi(vr(0).s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.usmmla(vr(0).s4(), vr(1).b16(), vr(2).b16());
            });
        snprintf(name, sizeof(name), "USMMLA v4s latency    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

void run_i8mm_tests(const BenchmarkParams& base_params) {
    run_i8mm_section(base_params, base_params.loops,
                     base_params.instructions_per_loop);
}

} // namespace arm64bench::gen
