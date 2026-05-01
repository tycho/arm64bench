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
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#elif defined(_WIN32)
#  include <windows.h>
#endif

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
                [](a64::Assembler& a) {
                    a.fmov(d_src(), 1.0);
                    a.dup(vd2_src(), d16.d(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(dr(i), 1.0);
                        a.dup(vd2(i), kDRegs[i].d(0));
                    }
                },
                [](a64::Assembler& a, uint32_t u) {
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
                [](a64::Assembler& a) {
                    a.fmov(dr(nc),     1.5); a.dup(vd2(nc),     kDRegs[nc].d(0));
                    a.fmov(dr(nc + 1), 2.0); a.dup(vd2(nc + 1), kDRegs[nc+1].d(0));
                    for (uint32_t i = 0; i < nc; ++i) {
                        a.fmov(dr(i), 1.0);
                        a.dup(vd2(i), kDRegs[i].d(0));
                    }
                },
                [](a64::Assembler& a, uint32_t u) {
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
                [](a64::Assembler& a) {
                    a.movi(vs4(nc),     Imm(3));
                    a.movi(vs4(nc + 1), Imm(7));
                    for (uint32_t i = 0; i < nc; ++i) a.movi(vs4(i), Imm(1));
                },
                [](a64::Assembler& a, uint32_t u) {
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
// Section 7: Cross-domain latency (GPR ↔ FP/SIMD register file)
// ════════════════════════════════════════════════════════════════════════════
//
// These tests measure the latency penalty for moving values between the
// integer (GPR) and floating-point/SIMD execution domains.
//
// On all ARM64 microarchitectures, GPR and SIMD/FP registers are physically
// separate files with dedicated execution ports. Moving a value from one to
// the other crosses an interconnect that has non-zero latency.
//
// FMOV Dn, Xn — copies integer bits from a GPR into the lower 64 bits of
//   a SIMD/FP register (no conversion; the bits are reinterpreted as FP only
//   by subsequent FP instructions).
//
// FMOV Xn, Dn — reverse direction.
//
// SCVTF Dn, Xn — convert a signed 64-bit integer in Xn to IEEE-754 double
//   precision in Dn. This both crosses the domain and performs a conversion.
//
// FCVTZS Xn, Dn — convert a double-precision float in Dn to a signed 64-bit
//   integer in Xn with truncation toward zero.
//
// ── Measurement approach ──────────────────────────────────────────────────
//
// Because each FMOV produces a result in a different domain from its input,
// a single FMOV cannot be chained with itself. Instead, tests use round-trip
// pairs: two complementary instructions that form a full dependency chain
// through both domains. With unroll=2N alternating instructions:
//
//   GPR→FP pair:   FMOV D0, X0 → FMOV X0, D0 (repeated N times)
//   Conv pair:     SCVTF D0, X0 → FCVTZS X0, D0 (repeated N times)
//
// The reported clk/insn is the average latency per instruction in the pair.
// If both directions have equal latency L, clk/insn = L.
//
// ── Architecture differentiation ─────────────────────────────────────────
//
// Apple M-series: domain crossing via FMOV is reportedly 0–1 extra cycles
//   over baseline. The forwarding network is tightly integrated.
//
// Cortex-A76/A78/X1: typically 0–2 cycle domain crossing overhead.
//
// Qualcomm Oryon: characteristics not yet published.

static void run_crossdomain_tests(const BenchmarkParams& base,
                                   uint64_t loops, uint32_t unroll) {
    char name[80];

    // Ensure even unroll: tests use pairs of complementary instructions.
    const uint32_t u2 = (unroll >= 2) ? (unroll / 2) * 2 : 2;

    // ── FMOV round-trip: GPR → FP → GPR ──────────────────────────────────
    // Chain: X0 → (FMOV D0,X0) → D0 → (FMOV X0,D0) → X0 → ...
    // clk/insn = avg(latency_GPR→FP, latency_FP→GPR).
    {
        auto fn = build_fp_loop(loops, u2,
            [](a64::Assembler& a) {
                // Seed: a 64-bit pattern that reads back as 1.0 in double.
                a.mov(x0, Imm(0x3FF0000000000000LL));
                a.fmov(d0, x0);
            },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.fmov(x0, d0);   // FP → GPR
                else       a.fmov(d0, x0);   // GPR → FP
            });
        snprintf(name, sizeof(name),
                 "FMOV GPR↔FP round-trip (%ux)", u2);
        run_one(name, fn, make_params(base, loops, u2));
    }

    // ── SCVTF / FCVTZS round-trip: integer → double → integer ────────────
    // Chain: X0 → (SCVTF D0,X0) → D0 → (FCVTZS X0,D0) → X0 → ...
    // Initial value X0 = 1 is stable: 1 → 1.0 → 1 → 1.0 → ...
    // clk/insn = avg(latency_SCVTF, latency_FCVTZS).
    {
        auto fn = build_fp_loop(loops, u2,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(1));
                a.scvtf(d0, x0);   // seed d0 = 1.0
            },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.fcvtzs(x0, d0);  // double → int64 (truncate)
                else       a.scvtf (d0, x0);  // int64 → double
            });
        snprintf(name, sizeof(name),
                 "SCVTF/FCVTZS d64 round-trip (%ux)", u2);
        run_one(name, fn, make_params(base, loops, u2));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 8: Cryptography extensions (ARMv8-A)
// ════════════════════════════════════════════════════════════════════════════
//
// ARMv8-A Cryptography extensions provide single-instruction acceleration for:
//   AES:    AESE/AESD (round), AESMC/AESIMC (MixColumns)
//   SHA-256: SHA256H, SHA256H2 (compression), SHA256SU0/SU1 (message schedule)
//   Poly multiply: PMULL/PMULL2 (poly8×8→16 lanes, or poly64×64→128 for GCM)
//   CRC32:  CRC32B/H/W/X, CRC32CB/CH/CW/CX (Castagnoli variant)
//
// All Apple Silicon, Snapdragon X Elite, and ARMv8.1+ Linux targets support
// these extensions. Instructions are JIT-emitted; no compile-time guards needed.
//
// ── AES microarchitecture notes ──────────────────────────────────────────
//
// Typical ARM cores fuse consecutive AESE+AESMC (and AESD+AESIMC) pairs on
// the same register into a single micro-op, reducing the pair to 1 cycle.
// Apple M-series: 2 AES execution units, each can execute 1 fused pair/cycle.
// At 2 independent AES streams, throughput saturates: 1 round pair/cycle total.
//
// ── PMULL poly64 (GCM) ───────────────────────────────────────────────────
//
// GCM (Galois/Counter Mode) authentication uses PMULL Vd.1Q, Vn.1D, Vm.1D
// to compute a 128-bit carry-less multiply. Latency on Apple M-series ~2 cyc.
//
// Chaining: Vd.1Q (128-bit output) → Vd.1D (lower 64 bits as next input).
//
// ── SHA-256 notes ─────────────────────────────────────────────────────────
//
// SHA256H Q0, Q1, V2.4S implements one round of SHA-256 compression.
// Q0 holds the first half of {a,b,c,d,e,f,g,h}; Q1 holds the second half.
// Both are read and written (or written via SHA256H2). Only Q0 is chained.

static void run_crypto_tests(const BenchmarkParams& base,
                              uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── AESE latency ──────────────────────────────────────────────────────
    // AESE V0.16B, V1.16B: V0 ← SubBytes(ShiftRows(V0)) XOR V1
    // V1 = constant round key. Chain through V0.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[1].b16(), Imm(0x5A));  // constant round key
                a.movi(kVRegs[0].b16(), Imm(0x01));  // data
            },
            [](a64::Assembler& a, uint32_t) {
                a.aese(kVRegs[0].b16(), kVRegs[1].b16());
            });
        snprintf(name, sizeof(name), "AESE latency          (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── AESMC latency ─────────────────────────────────────────────────────
    // AESMC V0.16B, V0.16B: V0 ← MixColumns(V0)
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) { a.movi(kVRegs[0].b16(), Imm(0x01)); },
            [](a64::Assembler& a, uint32_t) {
                a.aesmc(kVRegs[0].b16(), kVRegs[0].b16());
            });
        snprintf(name, sizeof(name), "AESMC latency         (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── AESE+AESMC pair latency ───────────────────────────────────────────
    // One AES-128 encryption round: AESE immediately followed by AESMC on
    // the same register. Hardware may fuse this into 1 micro-op.
    // Emits unroll/2 pairs = unroll instructions.
    {
        const uint32_t u2 = (unroll / 2) * 2;
        auto fn = build_fp_loop(loops, u2,
            [](a64::Assembler& a) {
                a.movi(kVRegs[1].b16(), Imm(0x5A));  // round key
                a.movi(kVRegs[0].b16(), Imm(0x01));  // data
            },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.aesmc(kVRegs[0].b16(), kVRegs[0].b16());
                else       a.aese (kVRegs[0].b16(), kVRegs[1].b16());
            });
        snprintf(name, sizeof(name), "AESE+AESMC latency    (%ux unroll)", u2);
        run_one(name, fn, make_params(base, loops, u2));
    }

    // ── AESE+AESMC throughput ─────────────────────────────────────────────
    // N independent AES data streams, all using the same constant key V(n).
    // Each stream: AESE Vi.16B, Vkey.16B → AESMC Vi.16B, Vi.16B
    // Reveals number of AES execution units (saturation chain count).
    {
        static const uint32_t kChains[] = { 2, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t key_reg = nc;
            const uint32_t u2 = nc * 2;  // 2 instructions per stream
            auto fn = build_fp_loop(loops, u2,
                [nc, key_reg](a64::Assembler& a) {
                    a.movi(kVRegs[key_reg].b16(), Imm(0x5A));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].b16(), Imm(static_cast<uint64_t>(i + 1)));
                },
                [key_reg](a64::Assembler& a, uint32_t u) {
                    const uint32_t chain = u / 2;
                    if (u & 1) a.aesmc(kVRegs[chain].b16(), kVRegs[chain].b16());
                    else       a.aese (kVRegs[chain].b16(), kVRegs[key_reg].b16());
                });
            snprintf(name, sizeof(name), "AESE+AESMC tput (%u streams, %ux)", nc, u2);
            run_one(name, fn, make_params(base, loops, u2));
        }
    }

    // ── PMULL poly64 latency (GCM form: 64×64 → 128) ─────────────────────
    // PMULL V0.1Q, V0.1D, V1.1D
    // Chain: V0.1D (lower 64 bits of V0) → V0.1Q (full 128-bit result).
    // V1 = constant multiplier (analogous to GCM authentication key H).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[1].b16(), Imm(0x03));   // constant multiplier
                a.movi(kVRegs[0].b16(), Imm(0xAA));   // data
            },
            [](a64::Assembler& a, uint32_t) {
                a.pmull(kVRegs[0].q(), kVRegs[0].d(), kVRegs[1].d());
            });
        snprintf(name, sizeof(name), "PMULL poly64 latency  (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── SHA256SU0 latency ─────────────────────────────────────────────────
    // SHA256SU0 V0.4S, V1.4S — message schedule step 0. Chains through V0.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[1].s4(), Imm(0x5A));
                a.movi(kVRegs[0].s4(), Imm(0x01));
            },
            [](a64::Assembler& a, uint32_t) {
                a.sha256su0(kVRegs[0].s4(), kVRegs[1].s4());
            });
        snprintf(name, sizeof(name), "SHA256SU0 latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── SHA256H latency ───────────────────────────────────────────────────
    // SHA256H Q0, Q1, V2.4S — compression round A. Q0 = f(Q0, Q1, V2.4S).
    // Q1 (second state half) and V2 (message words) held constant.
    // Chain through Q0 (first state half: a, b, c, d).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].s4(),  Imm(0x5A));   // message words W[t..t+3]
                a.movi(kVRegs[1].b16(), Imm(0x03));   // second state half (constant)
                a.movi(kVRegs[0].b16(), Imm(0x01));   // first state half (chains)
            },
            [](a64::Assembler& a, uint32_t) {
                a.sha256h(kVRegs[0].q(), kVRegs[1].q(), kVRegs[2].s4());
            });
        snprintf(name, sizeof(name), "SHA256H latency       (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CRC32B latency ────────────────────────────────────────────────────
    // CRC32B W0, W0, W1 — CRC-32 of byte W1[7:0], accumulated in W0.
    // W0 chains (CRC state). W1 = constant data byte.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xAB));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32b(w0, w0, w1); });
        snprintf(name, sizeof(name), "CRC32B latency        (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CRC32W latency ────────────────────────────────────────────────────
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xABCD1234));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32w(w0, w0, w1); });
        snprintf(name, sizeof(name), "CRC32W latency        (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CRC32X latency ────────────────────────────────────────────────────
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xABCD123456789ABCULL));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32x(w0, w0, x1); });
        snprintf(name, sizeof(name), "CRC32X latency        (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 9: Advanced SIMD — dot product, widening multiply, and FP16
// ════════════════════════════════════════════════════════════════════════════
//
// ── SDOT / UDOT (ARMv8.2-A DotProd extension) ────────────────────────────
//
// SDOT Vd.4S, Vn.16B, Vm.16B performs 4 independent signed int8 dot
// products of 4-element groups:
//   Vd[i] += Vn[4i..4i+3] · Vm[4i..4i+3]   (for i in 0..3)
//
// This is the primary instruction for quantized neural network inference
// on ARM (analogous to VNNI on x86). On Apple M-series and Snapdragon X1,
// it typically delivers 1 instruction per cycle throughput with multiple units.
//
// ── SMLAL / UMLAL (widening multiply-accumulate) ──────────────────────────
//
// SMLAL Vd.4S, Vn.4H, Vm.4H — signed 16×16 → 32-bit widening multiply-add.
//   Vd[i] += Vn[i] * Vm[i]  (for i in 0..3, inputs 16-bit, acc 32-bit)
//
// Used in fixed-point DSP / audio codecs. On modern Apple Silicon it likely
// shares the same MAC pipeline as FMLA.
//
// ── FMLA 8×f16 (FP16 FMA) ────────────────────────────────────────────────
//
// FMLA Vd.8H, Vn.8H, Vm.8H — fp16 fused multiply-accumulate (8 lanes).
// Requires __ARM_FEATURE_FP16_VECTOR_ARITHMETIC.
//
// Apple M-series supports fp16 natively in the FP pipeline. If f16 and f32
// FMA share the same units (same latency / same saturation count), then
// fp16 doubles the FLOPS throughput of f32 for the same pipeline width.
//
// ── FMLAL 4S (FP16 multiply → FP32 accumulate) ───────────────────────────
//
// FMLAL Vd.4S, Vn.4H, Vm.4H — multiply 4 fp16 pairs, accumulate to f32.
// Used in mixed-precision ML (compute in fp16, accumulate in fp32 to avoid
// overflow). Requires __ARM_FEATURE_FP16_FML.

static void run_advanced_simd_tests(const BenchmarkParams& base,
                                     uint64_t loops, uint32_t unroll) {
    char name[80];

#if defined(__ARM_FEATURE_DOTPROD)

    // ── SDOT v4s latency ──────────────────────────────────────────────────
    // SDOT V0.4S, V1.16B, V2.16B — V0 is the accumulator (chains).
    // V1 and V2 are constant signed-byte data.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x02));
                a.movi(kVRegs[1].b16(), Imm(0x03));
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.sdot(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "SDOT v4s latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── SDOT v4s throughput: sweep 2..6 chains ────────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            const uint32_t va = nc, vb = nc + 1;
            auto fn = build_fp_loop(loops, au,
                [nc, va, vb](a64::Assembler& a) {
                    a.movi(kVRegs[vb].b16(), Imm(0x02));
                    a.movi(kVRegs[va].b16(), Imm(0x03));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].s4(), Imm(0));
                },
                [nc, va, vb](a64::Assembler& a, uint32_t u) {
                    a.sdot(kVRegs[u % nc].s4(), kVRegs[va].b16(), kVRegs[vb].b16());
                });
            snprintf(name, sizeof(name),
                     "SDOT v4s tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── UDOT v4s latency (unsigned) ───────────────────────────────────────
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x02));
                a.movi(kVRegs[1].b16(), Imm(0x03));
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.udot(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "UDOT v4s latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

#endif // __ARM_FEATURE_DOTPROD

    // ── SMLAL v4s latency (int16×int16 → int32 widening accumulate) ───────
    // SMLAL V0.4S, V1.4H, V2.4H — V0 chains; V1, V2 constant (16-bit int)
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].h4(), Imm(0x03));
                a.movi(kVRegs[1].h4(), Imm(0x07));
                a.movi(kVRegs[0].s4(), Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.smlal(kVRegs[0].s4(), kVRegs[1].h4(), kVRegs[2].h4());
            });
        snprintf(name, sizeof(name), "SMLAL v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

    // ── FMLA 8×f16 latency ────────────────────────────────────────────────
    // FMLA V0.8H, V1.8H, V2.8H — fp16 FMA (8 lanes). V0 accumulator chains.
    // Init all regs via MOVI with 0x3C, LSL #8 → 0x3C00 = fp16(1.0).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].h8(), Imm(0x3C), Imm(8));  // fp16(1.0) in all lanes
                a.movi(kVRegs[1].h8(), Imm(0x3C), Imm(8));
                a.movi(kVRegs[0].h8(), Imm(0x3C), Imm(8));
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmla(kVRegs[0].h8(), kVRegs[1].h8(), kVRegs[2].h8());
            });
        snprintf(name, sizeof(name), "FMLA v8f16 acc-chain  (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── FMLA 8×f16 throughput: 4 chains ──────────────────────────────────
    {
        const uint32_t nc = 4;
        const uint32_t au = (unroll / nc) * nc;
        if (au) {
            auto fn = build_fp_loop(loops, au,
                [](a64::Assembler& a) {
                    a.movi(kVRegs[nc    ].h8(), Imm(0x3C), Imm(8));
                    a.movi(kVRegs[nc + 1].h8(), Imm(0x3C), Imm(8));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].h8(), Imm(0x3C), Imm(8));
                },
                [](a64::Assembler& a, uint32_t u) {
                    a.fmla(kVRegs[u % nc].h8(), kVRegs[nc].h8(), kVRegs[nc + 1].h8());
                });
            snprintf(name, sizeof(name),
                     "FMLA v8f16 tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__ARM_FEATURE_FP16_FML)

    // ── FMLAL v4s latency (f16×f16 → f32 widening accumulate) ────────────
    // FMLAL V0.4S, V1.4H, V2.4H — V0.4S is the f32 accumulator (chains).
    // V1 and V2 are fp16 inputs. Useful for mixed-precision ML inference.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].h4(), Imm(0x3C), Imm(8));  // fp16(1.0) in 4 lanes
                a.movi(kVRegs[1].h4(), Imm(0x3C), Imm(8));
                a.movi(kVRegs[0].s4(), Imm(0));              // f32(0.0) accumulator
            },
            [](a64::Assembler& a, uint32_t) {
                a.fmlal(kVRegs[0].s4(), kVRegs[1].h4(), kVRegs[2].h4());
            });
        snprintf(name, sizeof(name), "FMLAL v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

#endif // __ARM_FEATURE_FP16_FML
}

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

#if defined(__APPLE__)
static bool has_feat_i8mm() {
    int val = 0; size_t len = sizeof(val);
    return sysctlbyname("hw.optional.arm.FEAT_I8MM", &val, &len, nullptr, 0) == 0 && val != 0;
}
#elif defined(_WIN32)
static bool has_feat_i8mm() {
#  ifdef PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE
    return IsProcessorFeaturePresent(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
#  else
    return true;  // Snapdragon X Elite / Oryon always has FEAT_I8MM
#  endif
}
#else
static bool has_feat_i8mm() { return true; }
#endif

static void run_i8mm_tests(const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    if (!has_feat_i8mm()) {
        printf("  (FEAT_I8MM not available on this CPU — skipping)\n");
        return;
    }

    char name[80];

    // ── USDOT v4s latency ─────────────────────────────────────────────────
    // USDOT V0.4S, V1.16B, V2.16B — unsigned(V1) · signed(V2) dot product.
    // V1 = constant unsigned bytes, V2 = constant signed bytes. V0 chains.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x03));
                a.movi(kVRegs[1].b16(), Imm(0x02));
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.usdot(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "USDOT v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── USDOT v4s throughput: sweep 2..6 chains ───────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            const uint32_t va = nc, vb = nc + 1;
            auto fn = build_fp_loop(loops, au,
                [nc, va, vb](a64::Assembler& a) {
                    a.movi(kVRegs[vb].b16(), Imm(0x03));
                    a.movi(kVRegs[va].b16(), Imm(0x02));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].s4(), Imm(0));
                },
                [nc, va, vb](a64::Assembler& a, uint32_t u) {
                    a.usdot(kVRegs[u % nc].s4(), kVRegs[va].b16(), kVRegs[vb].b16());
                });
            snprintf(name, sizeof(name),
                     "USDOT v4s tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── SMMLA v4s latency ─────────────────────────────────────────────────
    // SMMLA V0.4S, V1.16B, V2.16B — 2×8 signed × 8×2 signed matrix MLA.
    // Each of the 4 int32 accumulators sums 8 int8×int8 products (vs 4 for SDOT).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x02));
                a.movi(kVRegs[1].b16(), Imm(0x03));
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.smmla(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "SMMLA v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── SMMLA v4s throughput: sweep 2..6 chains ───────────────────────────
    {
        static const uint32_t kChains[] = { 2, 3, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            const uint32_t va = nc, vb = nc + 1;
            auto fn = build_fp_loop(loops, au,
                [nc, va, vb](a64::Assembler& a) {
                    a.movi(kVRegs[vb].b16(), Imm(0x02));
                    a.movi(kVRegs[va].b16(), Imm(0x03));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].s4(), Imm(0));
                },
                [nc, va, vb](a64::Assembler& a, uint32_t u) {
                    a.smmla(kVRegs[u % nc].s4(), kVRegs[va].b16(), kVRegs[vb].b16());
                });
            snprintf(name, sizeof(name),
                     "SMMLA v4s tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── UMMLA v4s latency ─────────────────────────────────────────────────
    // UMMLA V0.4S, V1.16B, V2.16B — unsigned × unsigned matrix MLA.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x02));
                a.movi(kVRegs[1].b16(), Imm(0x03));
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.ummla(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "UMMLA v4s latency     (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── USMMLA v4s latency ────────────────────────────────────────────────
    // USMMLA V0.4S, V1.16B, V2.16B — unsigned(V1) × signed(V2) matrix MLA.
    // Matrix-multiply form of USDOT: 2× MAC depth per instruction.
    // The key instruction for INT8 quantized GEMM with asymmetric quantization.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[2].b16(), Imm(0x03));   // signed weights
                a.movi(kVRegs[1].b16(), Imm(0x02));   // unsigned activations
                a.movi(kVRegs[0].s4(),  Imm(0));
            },
            [](a64::Assembler& a, uint32_t) {
                a.usmmla(kVRegs[0].s4(), kVRegs[1].b16(), kVRegs[2].b16());
            });
        snprintf(name, sizeof(name), "USMMLA v4s latency    (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 11: Population count (NEON CNT) and emulation idioms
// ════════════════════════════════════════════════════════════════════════════
//
// ARM64 has no scalar POPCNT instruction. The idiomatic emulation of x86's
// POPCNT round-trips through NEON:
//
//     fmov  d0, x0        ; GPR → FP (cross-domain ~5 clk on Apple M)
//     cnt   v0.16b, v0.16b; per-byte popcount
//     addv  b0, v0.16b    ; horizontal sum across bytes (saturates a 6-bit total)
//     fmov  x0, d0        ; FP → GPR
//
// This is what x86→ARM64 emulators (Prism, Rosetta) emit for POPCNT, and
// what hand-written portable code uses when targeting ARM64. Measuring the
// end-to-end cost gives a realistic "POPCNT replacement" number.
//
// We also measure CTZ via the standard RBIT+CLZ idiom — ARM64 has no CTZ
// instruction either, but the two-step replacement is cheap (both 1 cyc).
//
// All instructions in this section (CNT, ADDV, FMOV, RBIT, CLZ) are
// baseline ARMv8.0-A. JIT-emitted; no runtime feature detection required.

static void run_popcount_idiom_tests(const BenchmarkParams& base,
                                     uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── NEON CNT v16b latency ─────────────────────────────────────────────
    // CNT V0.16B, V0.16B: per-byte popcount of v0 (16 lanes), result in v0.
    // Chain via v0 → v0. Operates on 8-bit lanes; output ∈ [0,8] per lane.
    // Initialised to 0x55 (popcount=4) so the chain converges to 0x03 (= 3,
    // popcount=2) and then 0x02 (popcount=1) and stabilises at 0x01 — all
    // valid working values that exercise the priority encoder.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(kVRegs[0].b16(), Imm(0x55));
            },
            [](a64::Assembler& a, uint32_t) {
                a.cnt(kVRegs[0].b16(), kVRegs[0].b16());
            });
        snprintf(name, sizeof(name), "CNT v16b latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── NEON CNT v16b throughput sweep ────────────────────────────────────
    // N independent CNT chains across v0..v(N-1).
    {
        static const uint32_t kCntChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kCntChains) {
            const uint32_t au = (unroll / nc) * nc;
            if (!au) continue;
            auto fn = build_fp_loop(loops, au,
                [nc](a64::Assembler& a) {
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(kVRegs[i].b16(), Imm(0x55));
                },
                [nc](a64::Assembler& a, uint32_t u) {
                    a.cnt(kVRegs[u % nc].b16(), kVRegs[u % nc].b16());
                });
            snprintf(name, sizeof(name),
                     "CNT v16b tput (%u chains, %ux unroll)", nc, au);
            run_one(name, fn, make_params(base, loops, au));
        }
    }

    // ── Scalar POPCNT emulation idiom (the x86 replacement) ───────────────
    // Four-instruction chain: GPR → FP → CNT → ADDV → FP → GPR.
    // Each unrolled iteration runs the full sequence; the chain runs through
    // x0, so each iteration's FMOV(d0,x0) sees the previous iteration's
    // popcount-sum result. Reported clk/insn = (sum of all 4 latencies) / 4.
    //
    // Apple M5 expectation: ~3–4 clk per insn (cross-domain FMOVs dominate;
    // CNT and ADDV are both ~2–3 cyc each). End-to-end ~14 clk per POPCNT
    // operation — illustrates why scalar POPCNT-heavy x86 code is so much
    // slower under emulation than native.
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0xDEADBEEFCAFEBABEULL));
            },
            [](a64::Assembler& a, uint32_t u) {
                switch (u & 3) {
                    case 0: a.fmov(d0, x0);                                 break;
                    case 1: a.cnt (kVRegs[0].b16(), kVRegs[0].b16());       break;
                    // ADDV Bd, Vn.16B — horizontal sum across all 16 bytes.
                    case 2: a.addv(b0, kVRegs[0].b16());                    break;
                    case 3: a.fmov(x0, d0);                                 break;
                }
            });
        snprintf(name, sizeof(name),
                 "POPCNT idiom (FMOV+CNT+ADDV+FMOV) (%ux)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CTZ (count trailing zeros) emulation idiom: RBIT then CLZ ─────────
    // ARM64 has no CTZ; the canonical replacement is RBIT (bit-reverse)
    // followed by CLZ. Both are baseline ARMv8.0 and typically 1 clk each,
    // so the two-instruction chain reports ~1 clk/insn (= ~2 clk per CTZ).
    {
        auto fn = build_fp_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x0, Imm(0xDEADBEEFCAFEBABEULL));
            },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.clz (x0, x0);
                else       a.rbit(x0, x0);
            });
        snprintf(name, sizeof(name),
                 "CTZ idiom (RBIT+CLZ chain) (%ux)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
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

    printf("\n── Cross-domain latency (GPR ↔ FP) ─────────────────────────────\n");
    run_crossdomain_tests(base_params, loops, unroll);

    printf("\n── Cryptography extensions ─────────────────────────────────────\n");
    run_crypto_tests(base_params, loops, unroll);

    printf("\n── Advanced SIMD (dot-product / widening / FP16) ───────────────\n");
    run_advanced_simd_tests(base_params, loops, unroll);

    printf("\n── FEAT_I8MM (int8 matrix multiply) ────────────────────────────\n");
    run_i8mm_tests(base_params, loops, unroll);

    printf("\n── Population count / POPCNT idiom ─────────────────────────────\n");
    run_popcount_idiom_tests(base_params, loops, unroll);
}

} // namespace arm64bench::gen
