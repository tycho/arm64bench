// gen_frontend.cpp
// Front-end and rename-stage probes. See gen_frontend.h.
//
// ── Reading the results ──────────────────────────────────────────────────────
//
// MOV elimination: a two-register ping-pong `mov x1, x0 ; mov x0, x1` is a
// true dependency chain. If the core executes it, each MOV costs 1 clk; if
// rename resolves it (register renaming aliasing), the chain costs ~0 clk
// and the loop runs at rename throughput. `add x0, x0, #0` alongside is the
// "definitely executed" 1-clk control.
//
// Zero idioms: `add x0, x0, x20 ; eor x0, x0, x0` alternating. If EOR is
// recognised as producing zero regardless of its input, the ADD depends
// only on a constant and the pair costs ~1 clk (0.5 clk/insn); if not, the
// pair is a real 2-instruction chain (1 clk/insn).
//
// Fusion: the pair test runs N independent (A;B) pairs per iteration and
// reports clk per *pair*. The two single tests report clk per instruction
// for A alone and B alone under the same register rotation. If pairs run at
// the rate of the cheaper single, the pair became one micro-op.

#include "gen_frontend.h"
#include "gen_common.h"
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── NOP / width ───────────────────────────────────────────────────────────────

static void run_nop_tests(const BenchmarkParams& base, uint64_t loops) {
    char name[80];
    // Small body: pure decode/rename width. Large bodies: the loop no longer
    // fits the loop buffer / fetch window, so I-cache fetch bandwidth shows.
    for (const uint32_t unroll : { 32u, 256u, 2048u, 8192u }) {
        const uint64_t l = (unroll > 256) ? loops / (unroll / 256) : loops;
        auto fn = build_loop(l, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { a.nop(); });
        snprintf(name, sizeof(name), "NOP tput            (%5ux unroll, %2u KB body)",
                 unroll, unroll * 4 / 1024);
        run_one(name, fn, params_for(base, l, unroll));
    }
}

// ── MOV elimination ───────────────────────────────────────────────────────────

static void run_mov_elim_tests(const BenchmarkParams& base, uint64_t loops, uint32_t unroll) {
    char name[80];
    const uint32_t u2 = chain_unroll(unroll, 1, 2);

    // Control: a chain the core must execute.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { a.mov(x0, Imm(1)); },
            [](a64::Assembler& a, uint32_t) { a.add(x0, x0, Imm(0)); });
        snprintf(name, sizeof(name), "ADD x0,x0,#0 chain  (%ux, executed control)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    // GPR ping-pong.
    {
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) { a.mov(x0, Imm(1)); a.mov(x1, Imm(2)); },
            [](a64::Assembler& a, uint32_t u) { if (u & 1) a.mov(x0, x1); else a.mov(x1, x0); });
        snprintf(name, sizeof(name), "MOV x1,x0;x0,x1 chain (%ux, elim → ~0 clk)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
    // GPR mov with a dependent ADD in the loop: mov x1,x0 ; add x0,x1,#1.
    // Eliminated MOV → 1 clk per pair; executed → 2 clk per pair.
    {
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) { a.mov(x0, Imm(1)); },
            [](a64::Assembler& a, uint32_t u) { if (u & 1) a.add(x0, x1, Imm(1)); else a.mov(x1, x0); });
        snprintf(name, sizeof(name), "MOV x1,x0;ADD x0,x1 chain (%ux, 2 insn/step)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
    // Scalar FP ping-pong (FMOV Dd, Dn).
    {
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) { a.fmov(d0, 1.0); a.fmov(d1, 2.0); },
            [](a64::Assembler& a, uint32_t u) { if (u & 1) a.fmov(d0, d1); else a.fmov(d1, d0); });
        snprintf(name, sizeof(name), "FMOV d1,d0;d0,d1 chain (%ux, elim → ~0 clk)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
    // Vector ping-pong (ORR Vd, Vn, Vn — the canonical vector MOV).
    {
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) { a.movi(v0.b16(), Imm(1)); a.movi(v1.b16(), Imm(2)); },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.orr(v0.b16(), v1.b16(), v1.b16());
                else       a.orr(v1.b16(), v0.b16(), v0.b16());
            });
        snprintf(name, sizeof(name), "ORR v1,v0;v0,v1 chain (%ux, elim → ~0 clk)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
}

// ── Zero idioms ───────────────────────────────────────────────────────────────

static void run_zero_idiom_tests(const BenchmarkParams& base, uint64_t loops, uint32_t unroll) {
    char name[80];
    const uint32_t u2 = chain_unroll(unroll, 1, 2);

    struct Idiom {
        const char* label;
        void (*zero)(a64::Assembler&);
        void (*add)(a64::Assembler&);
        void (*seed)(a64::Assembler&);
    };
    const Idiom idioms[] = {
        { "ADD;EOR x0,x0,x0",  [](a64::Assembler& a) { a.eor(x0, x0, x0); },
                               [](a64::Assembler& a) { a.add(x0, x0, x20); },
                               [](a64::Assembler& a) { a.mov(x0, Imm(1)); a.mov(x20, Imm(3)); } },
        { "ADD;SUB x0,x0,x0",  [](a64::Assembler& a) { a.sub(x0, x0, x0); },
                               [](a64::Assembler& a) { a.add(x0, x0, x20); },
                               [](a64::Assembler& a) { a.mov(x0, Imm(1)); a.mov(x20, Imm(3)); } },
        { "ADD;AND x0,x0,xzr", [](a64::Assembler& a) { a.and_(x0, x0, xzr); },
                               [](a64::Assembler& a) { a.add(x0, x0, x20); },
                               [](a64::Assembler& a) { a.mov(x0, Imm(1)); a.mov(x20, Imm(3)); } },
        { "ADD;EOR v0,v0,v0",  [](a64::Assembler& a) { a.eor(v0.b16(), v0.b16(), v0.b16()); },
                               [](a64::Assembler& a) { a.add(v0.s4(), v0.s4(), v1.s4()); },
                               [](a64::Assembler& a) { a.movi(v0.s4(), Imm(1)); a.movi(v1.s4(), Imm(3)); } },
        { "ADD;SUB v0,v0,v0",  [](a64::Assembler& a) { a.sub(v0.s4(), v0.s4(), v0.s4()); },
                               [](a64::Assembler& a) { a.add(v0.s4(), v0.s4(), v1.s4()); },
                               [](a64::Assembler& a) { a.movi(v0.s4(), Imm(1)); a.movi(v1.s4(), Imm(3)); } },
    };
    for (const Idiom& id : idioms) {
        auto zero = id.zero; auto add = id.add; auto seed = id.seed;
        auto fn = build_loop(loops, u2,
            [seed](a64::Assembler& a) { seed(a); },
            [zero, add](a64::Assembler& a, uint32_t u) { if (u & 1) zero(a); else add(a); });
        snprintf(name, sizeof(name), "%-19s (%ux; 0.5=idiom, 1.0=chain)", id.label, u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
}

// ── Macro-op fusion ───────────────────────────────────────────────────────────

static void run_fusion_tests(const BenchmarkParams& base, uint64_t loops, uint32_t unroll) {
    char name[80];
    static constexpr uint32_t kRot = 8;          // independent register rotation
    const uint32_t pairs = chain_unroll(unroll, kRot, 1);

    // ── CMP + B.cond (not taken) ──────────────────────────────────────────
    // x0 == x1 so B.NE never branches; each pair gets its own label so the
    // branch target is the next instruction.
    {
        auto fn = build_loop(loops, pairs,
            [](a64::Assembler& a) { a.mov(x0, Imm(5)); a.mov(x1, Imm(5)); },
            [](a64::Assembler& a, uint32_t) {
                Label l = a.new_label();
                a.cmp(xr(0), xr(1));
                a.b(CondCode::kNE, l);
                a.bind(l);
            });
        snprintf(name, sizeof(name), "CMP+B.NE pairs      (%u pairs/iter, clk per pair)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs,
            [](a64::Assembler& a) { a.mov(x0, Imm(5)); a.mov(x1, Imm(5)); },
            [](a64::Assembler& a, uint32_t) { a.cmp(xr(0), xr(1)); });
        snprintf(name, sizeof(name), "CMP alone           (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs,
            [](a64::Assembler& a) { a.mov(x0, Imm(5)); a.cmp(x0, x0); },   // flags: EQ
            [](a64::Assembler& a, uint32_t) {
                Label l = a.new_label();
                a.b(CondCode::kNE, l);
                a.bind(l);
            });
        snprintf(name, sizeof(name), "B.NE alone (not taken) (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }

    // ── ADRP + ADD ────────────────────────────────────────────────────────
    {
        auto fn = build_loop(loops, pairs, no_setup,
            [](a64::Assembler& a, uint32_t u) {
                Label here = a.new_label();
                a.bind(here);
                a.adrp(xr(u % kRot), here);
                a.add(xr(u % kRot), xr(u % kRot), Imm(0x123));
            });
        snprintf(name, sizeof(name), "ADRP+ADD pairs      (%u pairs/iter, clk per pair)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs, no_setup,
            [](a64::Assembler& a, uint32_t u) {
                Label here = a.new_label();
                a.bind(here);
                a.adrp(xr(u % kRot), here);
            });
        snprintf(name, sizeof(name), "ADRP alone          (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs,
            [](a64::Assembler& a) { for (uint32_t i = 0; i < kRot; ++i) a.mov(xr(i), Imm(i)); },
            [](a64::Assembler& a, uint32_t u) { a.add(xr(u % kRot), xr(u % kRot), Imm(0x123)); });
        snprintf(name, sizeof(name), "ADD imm alone       (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }

    // ── MOVZ + MOVK ───────────────────────────────────────────────────────
    {
        auto fn = build_loop(loops, pairs, no_setup,
            [](a64::Assembler& a, uint32_t u) {
                a.movz(xr(u % kRot), Imm(0x1234));
                a.movk(xr(u % kRot), Imm(0x5678), Imm(16));
            });
        snprintf(name, sizeof(name), "MOVZ+MOVK pairs     (%u pairs/iter, clk per pair)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs, no_setup,
            [](a64::Assembler& a, uint32_t u) { a.movz(xr(u % kRot), Imm(0x1234)); });
        snprintf(name, sizeof(name), "MOVZ alone          (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    {
        auto fn = build_loop(loops, pairs,
            [](a64::Assembler& a) { for (uint32_t i = 0; i < kRot; ++i) a.mov(xr(i), Imm(i)); },
            [](a64::Assembler& a, uint32_t u) { a.movk(xr(u % kRot), Imm(0x5678), Imm(16)); });
        snprintf(name, sizeof(name), "MOVK alone (dep on Xd) (%ux, clk per insn)", pairs);
        run_one(name, fn, params_for(base, loops, pairs));
    }
    // Four-instruction 64-bit immediate: the common `mov x, #imm64` expansion.
    {
        const uint32_t quads = chain_unroll(unroll, kRot, 1);
        auto fn = build_loop(loops, quads, no_setup,
            [](a64::Assembler& a, uint32_t u) {
                a.movz(xr(u % kRot), Imm(0x1111));
                a.movk(xr(u % kRot), Imm(0x2222), Imm(16));
                a.movk(xr(u % kRot), Imm(0x3333), Imm(32));
                a.movk(xr(u % kRot), Imm(0x4444), Imm(48));
            });
        snprintf(name, sizeof(name), "MOVZ+3xMOVK quads   (%u quads/iter, clk per quad)", quads);
        run_one(name, fn, params_for(base, loops, quads));
    }
}

// ── Macro-op fusion, rename-bound ─────────────────────────────────────────────
//
// Each group is one candidate pair followed by `pad` independent NOPs, so a
// group is pad + 2 instructions. NOPs cost nothing but a front-end slot, so
// the loop runs at the front-end's width: at 10 slots/clk a 10-instruction
// group costs 1.0 clk unfused and 0.9 clk if the pair takes one slot at
// whatever stage sets the width. Both pads are run because a 10 % step is
// small; the 20-instruction group shows the same absolute saving as a 5 %
// step, which tells a real fusion from a rounding wobble.
//
// A flat 1.0 for every pair, AESE+AESMC included (fused for latency, see
// gen_crypto), means the width limit is upstream of fusion: pairs fuse into
// one micro-op but still cost two fetch/decode slots. That is the M5 result.

static void run_fusion_rename_tests(const BenchmarkParams& base, uint64_t loops) {
    char name[96];
    static constexpr uint32_t kGroups = 32;
    static constexpr uint32_t kRot    = 8;

    struct Pair {
        const char* label;
        void (*emit)(a64::Assembler&, uint32_t);
    };
    static const Pair kPairs[] = {
        { "NOP+NOP  (control)",
          [](a64::Assembler& a, uint32_t) { a.nop(); a.nop(); } },
        { "CMP+B.NE (not taken)",
          [](a64::Assembler& a, uint32_t) {
              Label l = a.new_label();
              a.cmp(x0, x1); a.b(CondCode::kNE, l); a.bind(l); } },
        { "SUBS+B.NE (not taken)",
          [](a64::Assembler& a, uint32_t u) {
              Label l = a.new_label();
              a.subs(xr(2 + u % 6), xr(2 + u % 6), Imm(0)); a.b(CondCode::kEQ, l); a.bind(l); } },
        { "ADD+CBZ  (not taken)",
          [](a64::Assembler& a, uint32_t u) {
              Label l = a.new_label();
              a.add(xr(2 + u % 6), xr(2 + u % 6), Imm(0)); a.cbz(xr(2 + u % 6), l); a.bind(l); } },
        { "ADRP+ADD",
          [](a64::Assembler& a, uint32_t u) {
              Label here = a.new_label();
              a.bind(here);
              a.adrp(xr(u % kRot), here); a.add(xr(u % kRot), xr(u % kRot), Imm(0x123)); } },
        { "MOVZ+MOVK",
          [](a64::Assembler& a, uint32_t u) {
              a.movz(xr(u % kRot), Imm(0x1234)); a.movk(xr(u % kRot), Imm(0x5678), Imm(16)); } },
        { "AESE+AESMC",
          [](a64::Assembler& a, uint32_t u) {
              a.aese(vr(u % kRot).b16(), vr(8).b16()); a.aesmc(vr(u % kRot).b16(), vr(u % kRot).b16()); } },
    };

    for (const uint32_t pad : { 8u, 18u }) {
        for (const Pair& p : kPairs) {
            auto fn = build_loop(loops, kGroups,
                [](a64::Assembler& a) {
                    a.mov(x0, Imm(5)); a.mov(x1, Imm(5));
                    for (uint32_t i = 2; i < kRot; ++i) a.mov(xr(i), Imm(7));
                    for (uint32_t i = 0; i <= kRot; ++i) a.movi(vr(i).b16(), Imm(1));
                },
                [&p, pad](a64::Assembler& a, uint32_t u) {
                    p.emit(a, u);
                    for (uint32_t i = 0; i < pad; ++i) a.nop();
                });
            snprintf(name, sizeof(name), "%-22s + %2u NOP (clk per %2u-insn group)",
                     p.label, pad, pad + 2);
            run_one(name, fn, params_for(base, loops, kGroups));
        }
    }
}

// ── Branch throughput and ISB ─────────────────────────────────────────────────

static void run_branch_and_isb_tests(const BenchmarkParams& base, uint64_t loops, uint32_t unroll) {
    char name[80];

    // Taken unconditional branch to the next instruction.
    {
        auto fn = build_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) {
                Label l = a.new_label();
                a.b(l);
                a.bind(l);
            });
        snprintf(name, sizeof(name), "B taken (to next)   (%ux, clk per branch)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    // Taken B with an ALU op between: does a taken branch cost a fetch bubble?
    {
        const uint32_t u2 = chain_unroll(unroll, 1, 2);
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) { for (uint32_t i = 0; i < 8; ++i) a.mov(xr(i), Imm(i)); },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) { a.add(xr(u % 8), xr(u % 8), Imm(1)); }
                else       { Label l = a.new_label(); a.b(l); a.bind(l); }
            });
        snprintf(name, sizeof(name), "B taken + ADD alternating (%ux, clk per insn)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }
    // ISB: full pipeline flush.
    {
        const uint32_t u = 8;
        const uint64_t l = loops / 64;
        auto fn = build_loop(l, u, no_setup,
            [](a64::Assembler& a, uint32_t) { a.isb(Imm(15)); });
        snprintf(name, sizeof(name), "ISB                 (%ux, clk per ISB)", u);
        run_one(name, fn, params_for(base, l, u));
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

void run_frontend_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;

    section("Decode / rename width (NOP throughput)");
    run_nop_tests(base_params, loops);

    section("MOV elimination");
    run_mov_elim_tests(base_params, loops, unroll);

    section("Zero idioms (dependency breaking)");
    run_zero_idiom_tests(base_params, loops, unroll);

    section("Macro-op fusion (pairs/clk vs singles)");
    run_fusion_tests(base_params, loops, unroll);

    section("Macro-op fusion (rename-bound: pair + NOP padding)");
    run_fusion_rename_tests(base_params, loops);

    section("Branch throughput and ISB");
    run_branch_and_isb_tests(base_params, loops, unroll);
}

} // namespace arm64bench::gen
