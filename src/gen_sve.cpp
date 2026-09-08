// gen_sve.cpp
// SVE / SVE2 microbenchmarks. See gen_sve.h for the native vs streaming split.
//
// ── Registers ────────────────────────────────────────────────────────────────
//
// z8–z15 have callee-saved low halves (as d8–d15) and build_loop saves no
// vector state, so chains use z0–z7 and z16–z23; z24/z25 hold constants.
// p0 is the all-true governing predicate; p1–p3 are scratch for the
// predicate-throughput tests.
//
// ── Streaming mode ───────────────────────────────────────────────────────────
//
// SMSTART SM is emitted at the top of setup and SMSTOP SM in the teardown,
// so the streaming-mode entry/exit cost is paid once per timed call (tens of
// millions of instructions) and is invisible. Entering or leaving streaming
// mode zeroes every Z/P/V register — which is why every setup seeds its own
// registers, and why d8–d15 (callee-saved under AAPCS64) are saved to the
// loop's scratch area before SMSTART and restored after SMSTOP. Without that
// the C++ caller's values in d8–d15 are silently destroyed; the compiler
// happily keeps things like AsmJit operand signatures there across calls.
// Streaming mode forbids most NEON instructions; nothing in this file emits
// any between SMSTART and SMSTOP. The harness reference function (an ADD
// chain) runs outside the JIT'd function and is unaffected.

#include "gen_sve.h"
#include "gen_common.h"
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Mode ──────────────────────────────────────────────────────────────────────

enum class SveMode { None, Native, Streaming };

static SveMode detect_mode() {
    if (cpu_has(CpuFeature::SVE)) return SveMode::Native;
    if (cpu_has(CpuFeature::SME)) return SveMode::Streaming;
    return SveMode::None;
}

// ── Registers ─────────────────────────────────────────────────────────────────

static const ZReg kZ[] = { z0, z1, z2, z3, z4, z5, z6, z7,
                           z16, z17, z18, z19, z20, z21, z22, z23 };
static constexpr uint32_t kNumZ = sizeof(kZ) / sizeof(kZ[0]);
static const ZReg& zc0 = z24;   // constant operand
static const ZReg& zc1 = z25;   // second constant operand

// ── Builders ──────────────────────────────────────────────────────────────────

// Callee-saved FP registers, parked in the 64-byte scratch area around the
// streaming-mode section (SMSTART/SMSTOP zero them).
static constexpr uint32_t kFpSaveBytes = 64;

static void emit_sm_enter(a64::Assembler& a, bool sm) {
    if (!sm) return;
    a.stp(d8,  d9,  ptr(sp, 0));
    a.stp(d10, d11, ptr(sp, 16));
    a.stp(d12, d13, ptr(sp, 32));
    a.stp(d14, d15, ptr(sp, 48));
    a.smstart_sm();
}

static void emit_sm_leave(a64::Assembler& a, bool sm) {
    if (!sm) return;
    a.smstop_sm();
    a.ldp(d8,  d9,  ptr(sp, 0));
    a.ldp(d10, d11, ptr(sp, 16));
    a.ldp(d12, d13, ptr(sp, 32));
    a.ldp(d14, d15, ptr(sp, 48));
}

// build_loop with SMSTART/SMSTOP (and the d8–d15 save) around it in
// streaming mode.
template<class FSetup, class FBody>
static JitPool::TestFn build_sve_loop(SveMode mode, uint64_t loops, uint32_t unroll,
                                      FSetup&& setup, FBody&& body) {
    const bool sm = (mode == SveMode::Streaming);
    return build_loop_with_teardown(loops, unroll,
        [&](a64::Assembler& a) {
            emit_sm_enter(a, sm);
            a.ptrue(p0.s());
            setup(a);
        },
        body,
        [sm](a64::Assembler& a) { emit_sm_leave(a, sm); },
        kFpSaveBytes);
}

// Seed z regs 0..n-1 with a float pattern (all lanes) via a GPR broadcast.
static void seed_f32(a64::Assembler& a, uint32_t n, float value) {
    uint32_t bits; static_assert(sizeof(bits) == sizeof(value));
    __builtin_memcpy(&bits, &value, sizeof(bits));
    a.mov(w9, Imm(bits));
    for (uint32_t i = 0; i < n; ++i) a.dup(kZ[i].s(), w9);
}

static void seed_u32(a64::Assembler& a, uint32_t n, uint32_t value) {
    a.mov(w9, Imm(value));
    for (uint32_t i = 0; i < n; ++i) a.dup(kZ[i].s(), w9);
}

static void set_const_f32(a64::Assembler& a, const ZReg& z, float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    a.mov(w9, Imm(bits));
    a.dup(z.s(), w9);
}

// ── Vector length ─────────────────────────────────────────────────────────────

static uint64_t g_vl_bytes = 0;

static uint32_t probe_vector_length(SveMode mode) {
    g_vl_bytes = 0;
    const bool sm = (mode == SveMode::Streaming);
    auto fn = build_loop_with_teardown(1, 1,
        [sm](a64::Assembler& a) {
            emit_sm_enter(a, sm);
            a.mov(x1, Imm(reinterpret_cast<uint64_t>(&g_vl_bytes)));
        },
        [](a64::Assembler& a, uint32_t) {
            a.rdvl(x0, 1);            // x0 = vector length in bytes
            a.str(x0, ptr(x1));
        },
        [sm](a64::Assembler& a) { emit_sm_leave(a, sm); },
        kFpSaveBytes);
    if (!fn) return 0;
    fn();
    g_jit_pool->release(fn);
    return static_cast<uint32_t>(g_vl_bytes);
}

// ── Chain sweep on the SVE builder ───────────────────────────────────────────

template<class FSetup, class FBody>
static void sve_chain_sweep(SveMode mode, const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll, const char* prefix,
                            std::initializer_list<uint32_t> chains,
                            FSetup&& setup, FBody&& body) {
    char name[96];
    for (const uint32_t nc : chains) {
        const uint32_t au = chain_unroll(unroll, nc);
        auto fn = build_sve_loop(mode, loops, au,
            [&](a64::Assembler& a)             { setup(a, nc); },
            [&](a64::Assembler& a, uint32_t u) { body(a, nc, u); });
        snprintf(name, sizeof(name), "%s (%u chains, %ux unroll)", prefix, nc, au);
        run_one(name, fn, params_for(base, loops, au));
    }
}

// ── Sections ─────────────────────────────────────────────────────────────────

static void run_arith_tests(SveMode mode, const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    char name[96];

    // ADD z.s (integer, unpredicated)
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { seed_u32(a, 1, 1); a.mov(w9, Imm(3)); a.dup(zc0.s(), w9); },
            [](a64::Assembler& a, uint32_t) { a.add(z0.s(), z0.s(), zc0.s()); });
        snprintf(name, sizeof(name), "SVE ADD z.s latency   (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    sve_chain_sweep(mode, base, loops, unroll, "SVE ADD z.s tput", { 2, 4, 6, 8 },
        [](a64::Assembler& a, uint32_t nc) { seed_u32(a, nc, 1); a.mov(w9, Imm(3)); a.dup(zc0.s(), w9); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            const ZReg& z = kZ[u % nc]; a.add(z.s(), z.s(), zc0.s()); });

    // FADD z.s (unpredicated)
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { seed_f32(a, 1, 1.0f); set_const_f32(a, zc0, 1.0f); },
            [](a64::Assembler& a, uint32_t) { a.fadd(z0.s(), z0.s(), zc0.s()); });
        snprintf(name, sizeof(name), "SVE FADD z.s latency  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    sve_chain_sweep(mode, base, loops, unroll, "SVE FADD z.s tput", { 2, 4, 6, 8 },
        [](a64::Assembler& a, uint32_t nc) { seed_f32(a, nc, 1.0f); set_const_f32(a, zc0, 1.0f); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            const ZReg& z = kZ[u % nc]; a.fadd(z.s(), z.s(), zc0.s()); });

    // FMUL z.s (unpredicated): 1.0 × 1.0 stays 1.0, chain is real.
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { seed_f32(a, 1, 1.0f); set_const_f32(a, zc0, 1.0f); },
            [](a64::Assembler& a, uint32_t) { a.fmul(z0.s(), z0.s(), zc0.s()); });
        snprintf(name, sizeof(name), "SVE FMUL z.s latency  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // FMLA z.s (predicated, merging): z0 += zc0 * zc1, accumulator chains.
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) {
                seed_f32(a, 1, 0.0f); set_const_f32(a, zc0, 1.0f); set_const_f32(a, zc1, 0.5f); },
            [](a64::Assembler& a, uint32_t) { a.fmla(z0.s(), p0.m(), zc0.s(), zc1.s()); });
        snprintf(name, sizeof(name), "SVE FMLA z.s latency  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    sve_chain_sweep(mode, base, loops, unroll, "SVE FMLA z.s tput", { 2, 4, 6, 8 },
        [](a64::Assembler& a, uint32_t nc) {
            seed_f32(a, nc, 0.0f); set_const_f32(a, zc0, 1.0f); set_const_f32(a, zc1, 0.5f); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.fmla(kZ[u % nc].s(), p0.m(), zc0.s(), zc1.s()); });

    // SDOT z.s, z.b, z.b (accumulator chains; 4 × int8 MACs per lane)
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) {
                seed_u32(a, 1, 0); a.mov(w9, Imm(0x02020202)); a.dup(zc0.s(), w9);
                a.mov(w9, Imm(0x03030303)); a.dup(zc1.s(), w9); },
            [](a64::Assembler& a, uint32_t) { a.sdot(z0.s(), zc0.b(), zc1.b()); });
        snprintf(name, sizeof(name), "SVE SDOT z.s latency  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    sve_chain_sweep(mode, base, loops, unroll, "SVE SDOT z.s tput", { 2, 4, 6, 8 },
        [](a64::Assembler& a, uint32_t nc) {
            seed_u32(a, nc, 0); a.mov(w9, Imm(0x02020202)); a.dup(zc0.s(), w9);
            a.mov(w9, Imm(0x03030303)); a.dup(zc1.s(), w9); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.sdot(kZ[u % nc].s(), zc0.b(), zc1.b()); });

    // FADDV: horizontal reduction to a scalar, broadcast back to close the
    // chain. All-zero data keeps the values finite forever.
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { seed_f32(a, 1, 0.0f); },
            [](a64::Assembler& a, uint32_t) {
                a.faddv(s0, p0, z0.s());        // s0 = sum(z0) → lane 0 of z0
                a.dup(z0.s(), z0.s(0));         // broadcast lane 0 (2 insns/step)
            });
        snprintf(name, sizeof(name), "SVE FADDV+DUP chain   (%ux, 2 insn/step)", unroll);
        run_one(name, fn, params_for(base, loops, unroll * 2));
    }
}

alignas(64) static uint8_t g_ldst_buf[4096];

static void run_ldst_tests(SveMode mode, const BenchmarkParams& base,
                           uint64_t loops, uint32_t unroll, uint32_t vl_bytes) {
    char name[96];
    // 4 consecutive vectors at [x20 + k*VL]; 4 × 64 B = 256 B at SVL=512,
    // well inside the 4 KB buffer and L1 either way.
    static constexpr uint32_t nvec = 4;

    // LD1W {z.s}, p0/z, [x20, #k, MUL VL]
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { a.mov(x20, Imm(reinterpret_cast<uint64_t>(g_ldst_buf))); },
            [](a64::Assembler& a, uint32_t u) {
                a.ld1w(kZ[u % kNumZ].s(), p0.z(), ptr_vl(x20, static_cast<int32_t>(u % nvec)));
            });
        snprintf(name, sizeof(name), "SVE LD1W z.s L1 load  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll, vl_bytes));
    }
    // LDR z, [x20, #k, MUL VL] (unpredicated whole-vector load)
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { a.mov(x20, Imm(reinterpret_cast<uint64_t>(g_ldst_buf))); },
            [](a64::Assembler& a, uint32_t u) {
                a.ldr(kZ[u % kNumZ], ptr_vl(x20, static_cast<int32_t>(u % nvec)));
            });
        snprintf(name, sizeof(name), "SVE LDR z L1 load     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll, vl_bytes));
    }
    // ST1W {z.s}, p0, [x20, #k, MUL VL]
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x20, Imm(reinterpret_cast<uint64_t>(g_ldst_buf)));
                seed_u32(a, kNumZ, 0x11111111); },
            [](a64::Assembler& a, uint32_t u) {
                a.st1w(kZ[u % kNumZ].s(), p0, ptr_vl(x20, static_cast<int32_t>(u % nvec)));
            });
        snprintf(name, sizeof(name), "SVE ST1W z.s L1 store (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll, vl_bytes));
    }
    // STR z, [x20, #k, MUL VL]
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x20, Imm(reinterpret_cast<uint64_t>(g_ldst_buf)));
                seed_u32(a, kNumZ, 0x22222222); },
            [](a64::Assembler& a, uint32_t u) {
                a.str(kZ[u % kNumZ], ptr_vl(x20, static_cast<int32_t>(u % nvec)));
            });
        snprintf(name, sizeof(name), "SVE STR z L1 store    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll, vl_bytes));
    }
}

static void run_predicate_tests(SveMode mode, const BenchmarkParams& base,
                                uint64_t loops, uint32_t unroll) {
    char name[96];
    static const PReg kP[] = { p1, p2, p3 };

    // WHILELT p.s, w2, w3 — loop-control idiom; independent, rotating predicates.
    {
        auto fn = build_sve_loop(mode, loops, unroll,
            [](a64::Assembler& a) { a.mov(w2, Imm(0)); a.mov(w3, Imm(1000)); },
            [](a64::Assembler& a, uint32_t u) { a.whilelt(kP[u % 3].s(), w2, w3); });
        snprintf(name, sizeof(name), "SVE WHILELT p.s tput  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    // PTRUE p.s
    {
        auto fn = build_sve_loop(mode, loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t u) { a.ptrue(kP[u % 3].s()); });
        snprintf(name, sizeof(name), "SVE PTRUE p.s tput    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_sve_tests(const BenchmarkParams& base_params) {
    const SveMode mode = detect_mode();
    if (mode == SveMode::None) {
        printf("  (neither FEAT_SVE nor FEAT_SME available on this CPU — skipping SVE tests)\n");
        return;
    }

    const uint32_t vl_bytes = probe_vector_length(mode);
    if (vl_bytes == 0 || vl_bytes > 256) {
        printf("  (could not determine SVE vector length (RDVL gave %u) — skipping)\n", vl_bytes);
        return;
    }

    char title[96];
    snprintf(title, sizeof(title), "SVE (%s, VL = %u bits, %u × f32 lanes)",
             mode == SveMode::Native ? "native" : "streaming mode via SME",
             vl_bytes * 8, vl_bytes / 4);
    section(title);
    if (mode == SveMode::Streaming)
        printf("  Streaming SVE executes on the SME unit, not the core's NEON pipes;\n"
               "  SMSTART/SMSTOP bracket each timed call.\n");

    // Streaming-mode ops may be far slower than NEON; keep samples ~10–100 ms.
    const uint64_t loops  = scale_loops(1'500'000);
    const uint32_t unroll = base_params.instructions_per_loop;

    run_arith_tests(mode, base_params, loops, unroll);
    run_ldst_tests(mode, base_params, loops, unroll, vl_bytes);
    run_predicate_tests(mode, base_params, loops, unroll);
}

} // namespace arm64bench::gen
