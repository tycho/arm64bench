// gen_fp_simd.cpp
// Floating-point and NEON SIMD microbenchmark generator.
//
// ── Register conventions ─────────────────────────────────────────────────────
//
// ARM64 ABI for vector/FP registers:
//   v0–v7:   caller-saved  (we use as test chain registers)
//   v8–v15:  callee-saved  (not touched — we stay in v0–v7 and v16)
//   v16–v31: caller-saved  (we use v16 as the constant source)
//
// Our JIT functions save only x19 (loop counter) and x30 (LR).
// Vector registers v0–v7 and v16 are caller-saved; no prologue save needed.
//
// Prologue:  sub sp, sp, #16  /  stp x19, x30, [sp]
// Epilogue:  ldp x19, x30, [sp]  /  add sp, sp, #16  /  ret x30
//
// ── FMADD accumulator-chain test ─────────────────────────────────────────────
//
// For scalar integer MADD we observed a 1-cycle accumulator critical path:
// MADD with constant multiplicands lets the multiply run speculatively, and
// only the final addition is on the critical path.
//
// For FP FMADD, if the hardware uses a true fused multiply-add unit (not
// split add+mul), there is no separate "accumulator path" — the whole
// operation takes the full FMA latency (~4 cycles) regardless of which
// input is on the critical path. We test both to confirm.
//
// ── Slow instruction scaling ──────────────────────────────────────────────────
//
// FDIV / FSQRT: ~6–12 cycle latency. We use kSlowFpLoops × kSlowFpUnroll
// to keep each sample at ~100ms.

#include "gen_fp_simd.h"
#include "jit_buffer.h"
#include "harness.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Register tables ───────────────────────────────────────────────────────────

static const Vec kSRegs[17] = {
    s0,  s1,  s2,  s3,  s4,  s5,  s6,  s7,
    s8,  s9,  s10, s11, s12, s13, s14, s15, s16,
};
static const Vec kDRegs[17] = {
    d0,  d1,  d2,  d3,  d4,  d5,  d6,  d7,
    d8,  d9,  d10, d11, d12, d13, d14, d15, d16,
};
static const Vec kVRegs[17] = {
    v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,
    v8,  v9,  v10, v11, v12, v13, v14, v15, v16,
};

static inline const Vec& sr(uint32_t i)   { return kSRegs[i]; }
static inline const Vec& dr(uint32_t i)   { return kDRegs[i]; }
static inline Vec        vs4(uint32_t i)  { return kVRegs[i].s4(); }
static inline Vec        vd2(uint32_t i)  { return kVRegs[i].d2(); }

// Index 16 = constant source register (caller-saved, no save needed).
static inline const Vec& s_src()    { return kSRegs[16]; }   // s16
static inline const Vec& d_src()    { return kDRegs[16]; }   // d16
static inline Vec        vs4_src()  { return kVRegs[16].s4(); }
static inline Vec        vd2_src()  { return kVRegs[16].d2(); }

// ── Loop count helpers ────────────────────────────────────────────────────────

static constexpr uint64_t kSlowFpLoops  = 4'000'000;
static constexpr uint32_t kSlowFpUnroll = 8;

static BenchmarkParams make_params(const BenchmarkParams& base,
                                   uint64_t loops, uint32_t unroll) {
    BenchmarkParams p       = base;
    p.loops                 = loops;
    p.instructions_per_loop = unroll;
    p.bytes_per_insn        = 0;
    return p;
}

// ── Core loop builder ─────────────────────────────────────────────────────────
//
// emit_setup(a)     — called once after prologue; initialises all registers.
// emit_body(a, u)   — called `unroll` times; emits the instruction under test.

template<typename FSetup, typename FBody>
static JitPool::TestFn build_fp_loop(uint64_t loops, uint32_t unroll,
                                     FSetup&& emit_setup,
                                     FBody&&  emit_body) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(16));
    a.stp(x19, x30, ptr(sp));
    a.mov(x19, Imm(loops));

    emit_setup(a);

    a.align(AlignMode::kCode, 64);
    Label loop_top = a.new_label();
    a.bind(loop_top);

    for (uint32_t u = 0; u < unroll; ++u)
        emit_body(a, u);

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    a.ldp(x19, x30, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Benchmark runner helper ───────────────────────────────────────────────────

static void run_one(const char* name, JitPool::TestFn fn,
                    const BenchmarkParams& params) {
    if (!fn) return;
    benchmark(fn, name, params);
    g_jit_pool->release(fn);
}

// ════════════════════════════════════════════════════════════════════════════
// Section 1: Scalar f32 (single-precision)
// ════════════════════════════════════════════════════════════════════════════
//
// Expected M1 Firestorm:
//   FADD / FMUL latency:  3 cycles
//   FMADD latency:        4 cycles (true FMA, no shortcut accumulator path)
//   FDIV latency:         ~9–11 cycles (non-pipelined)
//   FSQRT latency:        ~8–10 cycles
//   FP add units:         4  (same ports handle FADD and FMUL)

static void run_scalar_f32_tests(const BenchmarkParams& base,
                                  uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── FADD f32 latency ──────────────────────────────────────────────────
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.0);
                a.fmov(sr(0),   1.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fadd(sr(0), sr(0), s_src());
            });
        snprintf(name, sizeof(name), "FADD f32 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FADD f32 throughput: sweep 2..8 chains ────────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(s_src(), 1.0);
                    for (uint32_t i = 0; i < nc; ++i) a.fmov(sr(i), 1.0);
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fadd(sr(u % nc), sr(u % nc), s_src());
                });
            snprintf(name, sizeof(name),
                     "FADD f32 tput  (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── FMUL f32 latency ──────────────────────────────────────────────────
    // Use 1.5 as constant to avoid exact 1.0 folding optimizations.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.5);
                a.fmov(sr(0),   1.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmul(sr(0), sr(0), s_src());
            });
        snprintf(name, sizeof(name), "FMUL f32 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FMUL f32 throughput: sweep 2..8 chains ────────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(s_src(), 1.5);
                    for (uint32_t i = 0; i < nc; ++i) a.fmov(sr(i), 1.0);
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fmul(sr(u % nc), sr(u % nc), s_src());
                });
            snprintf(name, sizeof(name),
                     "FMUL f32 tput  (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── FMADD f32 accumulator-chain latency ───────────────────────────────
    // s0 = s1*s2 + s0, with s1=1.5, s2=2.0 constant.
    // s1*s2 = 3.0 is computed speculatively. Critical path: through s0.
    // If FMADD has a short accumulator path: ~1–2 cycles (like integer MADD).
    // If true FMA unit (no shortcut):        ~4 cycles.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(sr(0), 1.0);  // accumulator (chains)
                a.fmov(sr(1), 1.5);  // multiplicand (stable)
                a.fmov(sr(2), 2.0);  // multiplier   (stable)
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmadd(sr(0), sr(1), sr(2), sr(0));  // s0 = s1*s2 + s0
            });
        snprintf(name, sizeof(name), "FMADD f32 acc-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FMADD f32 multiply-chain latency ──────────────────────────────────
    // s1 = s1*s2 + 0, with s2=1.5 stable, s0=0 constant.
    // Critical path: through s1. Expected: full FMADD latency (~4 cycles).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(sr(0), 0.0);  // constant zero accumulator
                a.fmov(sr(1), 1.5);  // multiplicand (chains)
                a.fmov(sr(2), 2.0);  // multiplier   (stable)
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmadd(sr(1), sr(1), sr(2), sr(0));  // s1 = s1*s2 + 0
            });
        snprintf(name, sizeof(name), "FMADD f32 mul-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FDIV f32 latency ──────────────────────────────────────────────────
    // Oscillating chain: s0 = 2.0/s0 alternates between 1.0 and 2.0.
    // 2.0 and 1.0 are both exactly representable and directly encodable via FMOV.
    {
        auto fn = build_fp_loop(kSlowFpLoops, kSlowFpUnroll,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 2.0);
                a.fmov(sr(0),   2.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fdiv(sr(0), s_src(), sr(0));  // s0 = 2.0/s0 (oscillates 1↔2)
            });
        snprintf(name, sizeof(name),
                 "FDIV f32 latency      (%ux unroll)", kSlowFpUnroll);
        run_one(name, fn, make_params(base, kSlowFpLoops, kSlowFpUnroll));
    }

    // ── FSQRT f32 latency ─────────────────────────────────────────────────
    // sqrt(2) ≈ 1.414, sqrt(1.414) ≈ 1.189, converges slowly toward 1.0.
    // Genuine dependency chain throughout.
    {
        auto fn = build_fp_loop(kSlowFpLoops, kSlowFpUnroll,
            [](a64::Assembler& a) { a.fmov(sr(0), 2.0); },
            [](a64::Assembler& a, uint32_t) { a.fsqrt(sr(0), sr(0)); });
        snprintf(name, sizeof(name),
                 "FSQRT f32 latency     (%ux unroll)", kSlowFpUnroll);
        run_one(name, fn, make_params(base, kSlowFpLoops, kSlowFpUnroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: Scalar f64 (double-precision)
// ════════════════════════════════════════════════════════════════════════════
//
// On most ARM64 cores, f64 and f32 share the same FP units with identical
// latency/throughput. Measurable differences indicate narrower f64 paths
// (common on in-order/mobile cores like Cortex-A55).

static void run_scalar_f64_tests(const BenchmarkParams& base,
                                  uint64_t loops, uint32_t unroll) {
    char name[80];

    // FADD f64 latency
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(d_src(), 1.0);
                a.fmov(dr(0),   1.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fadd(dr(0), dr(0), d_src());
            });
        snprintf(name, sizeof(name), "FADD f64 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // FADD f64 throughput: 2..8 chains
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(d_src(), 1.0);
                    for (uint32_t i = 0; i < nc; ++i) a.fmov(dr(i), 1.0);
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fadd(dr(u % nc), dr(u % nc), d_src());
                });
            snprintf(name, sizeof(name),
                     "FADD f64 tput  (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // FMUL f64 latency
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(d_src(), 1.5);
                a.fmov(dr(0),   1.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmul(dr(0), dr(0), d_src());
            });
        snprintf(name, sizeof(name), "FMUL f64 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // FDIV f64 latency
    {
        auto fn = build_fp_loop(kSlowFpLoops, kSlowFpUnroll,
            [](a64::Assembler& a) {
                a.fmov(d_src(), 2.0);
                a.fmov(dr(0),   2.0);
            },
            [](a64::Assembler& a, uint32_t) {
                a.fdiv(dr(0), d_src(), dr(0));
            });
        snprintf(name, sizeof(name),
                 "FDIV f64 latency      (%ux unroll)", kSlowFpUnroll);
        run_one(name, fn, make_params(base, kSlowFpLoops, kSlowFpUnroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: NEON vector 4×f32
// ════════════════════════════════════════════════════════════════════════════
//
// Each instruction operates on 4 f32 lanes in a 128-bit register.
// The harness counts each vector instruction as "1 insn", so clk/insn here
// is clk per 4-wide vector op (not per individual f32 element).
//
// KEY QUESTION: does vector FADD saturate at the same chain count as scalar
// FADD? If yes, scalar and vector FP share the same execution units.
//
// FMLA (vector fused multiply-accumulate):
//   vd.4s = vd.4s + vn.4s × vm.4s
//   This is the primary instruction for GEMM / convolution / neural network
//   inference. Its throughput (GFLOPS/s) = (4 ops × freq) / clk_per_insn.

static void run_neon_f32_tests(const BenchmarkParams& base,
                                uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── FADD 4×f32 latency ────────────────────────────────────────────────
    // Initialize v16.s4 as constant {1.0, 1.0, 1.0, 1.0} via scalar FMOV
    // then DUP to broadcast. v0.s4 is the chained accumulator.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.0);
                a.dup(vs4_src(), s16.s(0));   // v16.s4 = {1,1,1,1}
                a.fmov(sr(0), 1.0);
                a.dup(vs4(0), s0.s(0));        // v0.s4  = {1,1,1,1}
            },
            [](a64::Assembler& a, uint32_t) {
                a.fadd(vs4(0), vs4(0), vs4_src());
            });
        snprintf(name, sizeof(name), "FADD v4f32 latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FADD 4×f32 throughput: sweep 2..8 chains ─────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(s_src(), 1.0);
                    a.dup(vs4_src(), s16.s(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(sr(i), 1.0);
                        a.dup(vs4(i), kSRegs[i].s(0));
                    }
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fadd(vs4(u % nc), vs4(u % nc), vs4_src());
                });
            snprintf(name, sizeof(name),
                     "FADD v4f32 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── FMUL 4×f32 latency ────────────────────────────────────────────────
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.5);
                a.dup(vs4_src(), s16.s(0));
                a.fmov(sr(0), 1.0);
                a.dup(vs4(0), s0.s(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmul(vs4(0), vs4(0), vs4_src());
            });
        snprintf(name, sizeof(name), "FMUL v4f32 latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FMLA 4×f32 accumulator-chain latency ─────────────────────────────
    // v0 = v0 + v1×v2, with v1={1.5} and v2={2.0} stable constants.
    // Tests whether FMLA has a short accumulator forwarding path.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(sr(0), 1.0); a.dup(vs4(0), s0.s(0));  // acc
                a.fmov(sr(1), 1.5); a.dup(vs4(1), s1.s(0));  // mul A
                a.fmov(sr(2), 2.0); a.dup(vs4(2), s2.s(0));  // mul B
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmla(vs4(0), vs4(1), vs4(2));   // v0.4s += v1.4s × v2.4s
            });
        snprintf(name, sizeof(name), "FMLA v4f32 acc-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FMLA 4×f32 throughput: sweep 2..6 chains ─────────────────────────
    // Multiple independent accumulator chains. The multiplier pair (vm_a, vm_b)
    // requires two registers outside the accumulator range:
    //   nc=2: accumulators v0–v1, multipliers v2, v3
    //   nc=4: accumulators v0–v3, multipliers v4, v5
    //   nc=6: accumulators v0–v5, multipliers v6, v7
    //
    // MAX nc=6: at nc=8 the multiplier placement (capped to v6, v7) would
    // alias with accumulator chains 6 and 7, creating false write-after-read
    // dependencies that corrupt the measurement. 6 chains is the safe maximum
    // with 8 available vector registers (v0–v7).
    //
    // This is also sufficient to find the throughput floor: with 4-cycle
    // FMLA latency and 4 FP units, saturation occurs at N=16 chains —
    // well beyond our register budget. The 6-chain result gives the best
    // achievable throughput within this constraint.
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;

            // Multiplier registers sit just above the accumulator range.
            const uint32_t vm_a = nc;
            const uint32_t vm_b = nc + 1;

            auto fn = build_fp_loop(loops, au,
                [nc, vm_a, vm_b](a64::Assembler& a) {
                    a.fmov(sr(vm_a), 1.5); a.dup(vs4(vm_a), kSRegs[vm_a].s(0));
                    a.fmov(sr(vm_b), 2.0); a.dup(vs4(vm_b), kSRegs[vm_b].s(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(sr(i), 1.0);
                        a.dup(vs4(i), kSRegs[i].s(0));
                    }
                },
                [nc, vm_a, vm_b](a64::Assembler& a, uint32_t u) {
                    a.fmla(vs4(u % nc), vs4(vm_a), vs4(vm_b));
                });
            snprintf(name, sizeof(name),
                     "FMLA v4f32 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 4: NEON vector 2×f64
// ════════════════════════════════════════════════════════════════════════════
//
// Compares with 4×f32: if clk/insn is the same, the hardware processes
// 128 bits per cycle uniformly regardless of element precision.

static void run_neon_f64_tests(const BenchmarkParams& base,
                                uint64_t loops, uint32_t unroll) {
    char name[80];

    // FADD 2×f64 latency
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(d_src(), 1.0);
                a.dup(vd2_src(), d16.d(0));
                a.fmov(dr(0), 1.0);
                a.dup(vd2(0), d0.d(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.fadd(vd2(0), vd2(0), vd2_src());
            });
        snprintf(name, sizeof(name), "FADD v2f64 latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // FADD 2×f64 throughput: 4 chains
    {
        const uint32_t nc = 4;
        const uint32_t au = (unroll / nc) * nc;
        if (au) {
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(d_src(), 1.0);
                    a.dup(vd2_src(), d16.d(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(dr(i), 1.0);
                        a.dup(vd2(i), kDRegs[i].d(0));
                    }
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fadd(vd2(u % nc), vd2(u % nc), vd2_src());
                });
            snprintf(name, sizeof(name),
                     "FADD v2f64 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // FMLA 2×f64 latency
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.fmov(dr(0), 1.0); a.dup(vd2(0), d0.d(0));
                a.fmov(dr(1), 1.5); a.dup(vd2(1), d1.d(0));
                a.fmov(dr(2), 2.0); a.dup(vd2(2), d2.d(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmla(vd2(0), vd2(1), vd2(2));
            });
        snprintf(name, sizeof(name), "FMLA v2f64 acc-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // FMLA 2×f64 throughput: 4 chains
    {
        const uint32_t nc = 4;
        const uint32_t au = (unroll / nc) * nc;
        if (au) {
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.fmov(dr(nc),     1.5); a.dup(vd2(nc),     kDRegs[nc].d(0));
                    a.fmov(dr(nc + 1), 2.0); a.dup(vd2(nc + 1), kDRegs[nc+1].d(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(dr(i), 1.0);
                        a.dup(vd2(i), kDRegs[i].d(0));
                    }
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.fmla(vd2(u % nc), vd2(nc), vd2(nc + 1));
                });
            snprintf(name, sizeof(name),
                     "FMLA v2f64 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 5: Integer NEON 4×i32
// ════════════════════════════════════════════════════════════════════════════
//
// MOVI Vd.4s, #imm8 sets all 4 lanes of a 32-bit vector to the 8-bit
// zero-extended immediate. This is the standard way to initialise integer
// vector registers to small constants.
//
// Comparing with gen_integer scalar results:
//   If NEON ADD 4×i32 ≈ scalar ADD in clk/insn → NEON shares integer ALU.
//   If NEON ADD is slower → NEON integer has separate (fewer) ports.

static void run_neon_int_tests(const BenchmarkParams& base,
                                uint64_t loops, uint32_t unroll) {
    char name[80];

    // ADD 4×i32 latency
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vs4_src(), Imm(1));
                a.movi(vs4(0),    Imm(1));
            },
            [](a64::Assembler& a, uint32_t) {
                a.add(vs4(0), vs4(0), vs4_src());
            });
        snprintf(name, sizeof(name), "ADD  v4i32 latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ADD 4×i32 throughput: 2..8 chains
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.movi(vs4_src(), Imm(1));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(vs4(i), Imm(static_cast<uint64_t>(i + 1)));
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.add(vs4(u % nc), vs4(u % nc), vs4_src());
                });
            snprintf(name, sizeof(name),
                     "ADD  v4i32 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // MLA 4×i32 latency (accumulator chain: v0 += v1×v2)
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vs4(0), Imm(1));  // acc
                a.movi(vs4(1), Imm(3));  // mul A
                a.movi(vs4(2), Imm(7));  // mul B
            },
            [](a64::Assembler& a, uint32_t) {
                a.mla(vs4(0), vs4(1), vs4(2));
            });
        snprintf(name, sizeof(name), "MLA  v4i32 acc-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // MLA 4×i32 throughput: 4 chains
    {
        const uint32_t nc = 4;
        const uint32_t au = (unroll / nc) * nc;
        if (au) {
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    a.movi(vs4(nc),     Imm(3));
                    a.movi(vs4(nc + 1), Imm(7));
                    for (uint32_t i = 0; i < nc; ++i) a.movi(vs4(i), Imm(1));
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.mla(vs4(u % nc), vs4(nc), vs4(nc + 1));
                });
            snprintf(name, sizeof(name),
                     "MLA  v4i32 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 6: Mixed port pressure
// ════════════════════════════════════════════════════════════════════════════
//
// Interleaved scalar FP and NEON vector FP instructions test whether they
// compete for the same execution ports. On M1 Firestorm, scalar FP and NEON
// vector share the same FP pipeline — mixing them should show clear contention.

static void run_mixed_fp_tests(const BenchmarkParams& base,
                                uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── Scalar FADD + vector FADD interleaved (4 scalar + 2 vector chains) ─
    if (unroll >= 6) {
        const uint32_t au = (unroll / 6) * 6;
        auto fn = build_fp_loop(loops, au,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.0);
                a.dup(vs4_src(), s16.s(0));
                for (uint32_t i = 0; i < 4; ++i) a.fmov(sr(i), 1.0);
                a.fmov(sr(4), 1.0); a.dup(vs4(4), s4.s(0));
                a.fmov(sr(5), 1.0); a.dup(vs4(5), s5.s(0));
            },
            [](a64::Assembler& a, uint32_t u) {
                switch (u % 6) {
                    case 0: a.fadd(sr(0), sr(0), s_src());       break;
                    case 1: a.fadd(sr(1), sr(1), s_src());       break;
                    case 2: a.fadd(vs4(4), vs4(4), vs4_src());   break;
                    case 3: a.fadd(sr(2), sr(2), s_src());       break;
                    case 4: a.fadd(sr(3), sr(3), s_src());       break;
                    case 5: a.fadd(vs4(5), vs4(5), vs4_src());   break;
                    default: break;
                }
            });
        snprintf(name, sizeof(name),
                 "FADD scalar+vec mix  (4s+2v, %ux unroll)", au);
        run_one(name, fn, make_params(base, loops, au));
    }

    // ── Scalar FMUL + NEON FMLA interleaved (3 FMUL + 3 FMLA chains) ──────
    // FMUL and FMLA may use different sub-units; this probes port sharing
    // between plain multiply and fused multiply-accumulate.
    if (unroll >= 6) {
        const uint32_t au = (unroll / 6) * 6;
        // Layout: s0..s2 = scalar FMUL chains, v3..v5 = NEON FMLA chains
        //         s16 = scalar mult constant, v6 = vector mult A, v7 = mult B
        auto fn = build_fp_loop(loops, au,
            [](a64::Assembler& a) {
                a.fmov(s_src(), 1.5);
                for (uint32_t i = 0; i < 3; ++i) a.fmov(sr(i), 1.0);
                a.fmov(sr(6), 1.5); a.dup(vs4(6), s6.s(0));
                a.fmov(sr(7), 2.0); a.dup(vs4(7), s7.s(0));
                for (uint32_t i = 3; i < 6; ++i) {
                    a.fmov(sr(i), 1.0); a.dup(vs4(i), kSRegs[i].s(0));
                }
            },
            [](a64::Assembler& a, uint32_t u) {
                switch (u % 6) {
                    case 0: a.fmul(sr(0), sr(0), s_src());     break;
                    case 1: a.fmla(vs4(3), vs4(6), vs4(7));    break;
                    case 2: a.fmul(sr(1), sr(1), s_src());     break;
                    case 3: a.fmla(vs4(4), vs4(6), vs4(7));    break;
                    case 4: a.fmul(sr(2), sr(2), s_src());     break;
                    case 5: a.fmla(vs4(5), vs4(6), vs4(7));    break;
                    default: break;
                }
            });
        snprintf(name, sizeof(name),
                 "FMUL+FMLA mix        (3s+3v, %ux unroll)", au);
        run_one(name, fn, make_params(base, loops, au));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_fp_simd_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;

    printf("\n── Scalar f32 ──────────────────────────────────────────────────\n");
    run_scalar_f32_tests(base_params, loops, unroll);

    printf("\n── Scalar f64 ──────────────────────────────────────────────────\n");
    run_scalar_f64_tests(base_params, loops, unroll);

    printf("\n── NEON 4×f32 ──────────────────────────────────────────────────\n");
    run_neon_f32_tests(base_params, loops, unroll);

    printf("\n── NEON 2×f64 ──────────────────────────────────────────────────\n");
    run_neon_f64_tests(base_params, loops, unroll);

    printf("\n── NEON integer 4×i32 ──────────────────────────────────────────\n");
    run_neon_int_tests(base_params, loops, unroll);

    printf("\n── Mixed FP port pressure ──────────────────────────────────────\n");
    run_mixed_fp_tests(base_params, loops, unroll);
}

} // namespace arm64bench::gen
