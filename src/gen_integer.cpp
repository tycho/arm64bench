// gen_integer.cpp
// Integer ALU microbenchmark generator.
//
// ── Design notes ─────────────────────────────────────────────────────────────
//
// LATENCY vs THROUGHPUT
//   To measure *latency*, we chain each instruction so its output feeds
//   directly into the next instruction's input. This forces strictly serial
//   execution regardless of how many execution units exist.
//
//   To measure *throughput*, we run N independent chains (N separate result
//   registers, all reading from a constant source). The CPU can issue to
//   N execution units simultaneously. As we increase N, clk/insn drops
//   until we saturate the available ports — at which point adding more
//   chains produces no further improvement.
//
//   The gap between latency and throughput tells you two things:
//     1. The instruction's pipeline depth (latency in cycles).
//     2. How many execution units handle it (1/throughput_clk_per_insn).
//
// LOOP STRUCTURE (generated machine code)
//   sub  sp, sp, #16
//   stp  x19, x20, [sp]       // save callee-saved regs (x19=counter, x20=const)
//   mov  x19, #loops          // loop iteration count
//   mov  x20, #source_val     // constant operand (never modified by test)
//   mov  x0..xN, #init_vals   // seed test registers
//   align 64                  // align loop to cache line (consistent fetch)
// loop_top:
//   <test body, unroll times>
//   sub  x19, x19, #1
//   cbnz x19, loop_top        // SUB+CBNZ avoids writing NZCV (no flag deps)
//   ldp  x19, x20, [sp]
//   add  sp, sp, #16
//   ret
//
// WHY SUB + CBNZ (not SUBS + B.NE)?
//   SUBS writes the NZCV condition flags register. If the test body under
//   measurement also reads or writes NZCV (e.g. ADDS, SUBS, CMP), the
//   loop counter decrement would create a false dependency. SUB + CBNZ
//   is strictly cleaner: SUB never writes NZCV, and CBNZ reads the
//   register directly without touching NZCV at all.
//
// SLOW INSTRUCTION LOOP SCALING
//   Instructions like SDIV/UDIV have latencies of ~10–25 cycles, making
//   them 10–25× slower per iteration than ADD. Running them with the same
//   loop count as ADD would produce 9+ second samples. We scale loops down
//   proportionally so samples stay in the ~100ms range.

#include "gen_integer.h"
#include "jit_buffer.h"
#include "harness.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Register helpers ──────────────────────────────────────────────────────────
//
// AsmJit's newer a64 API collapses GpX/GpW into a single Gp type whose width
// is encoded in the register object itself (from the predefined constants).
// We use static lookup tables of those predefined constants so we never need
// to know the internal Gp constructor signature.
//
// Registers 0–15 are safe to use as scratch; x19 and x20 are reserved by
// build_loop for the loop counter and constant source respectively.

static const a64::Gp kXRegs[] = {
    x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7,
    x8,  x9,  x10, x11, x12, x13, x14, x15,
};
static const a64::Gp kWRegs[] = {
    w0,  w1,  w2,  w3,  w4,  w5,  w6,  w7,
    w8,  w9,  w10, w11, w12, w13, w14, w15,
};

static inline const a64::Gp& xr(uint32_t i) { return kXRegs[i]; }
[[maybe_unused]] static inline const a64::Gp& wr(uint32_t i) { return kWRegs[i]; }

// ── Loop configuration ────────────────────────────────────────────────────────

struct LoopConfig {
    uint64_t loops;             // iterations baked into the JIT'd loop
    uint32_t unroll;            // copies of the test body per iteration
    uint64_t source_val;        // value loaded into x20 (constant, never written)

    // Initial values for x0..x(num_init_regs-1).
    // Remaining registers (up to x15) are left at whatever build_loop sets.
    uint64_t init_vals[16];
    uint32_t num_init_regs;
};

// Sensible defaults: non-zero registers, a useful constant in x20.
static LoopConfig default_cfg(uint64_t loops, uint32_t unroll) {
    LoopConfig cfg{};
    cfg.loops        = loops;
    cfg.unroll       = unroll;
    cfg.source_val   = 0x12345678ULL;   // non-zero constant for x20
    cfg.num_init_regs = 8;
    for (uint32_t i = 0; i < 8; ++i)
        cfg.init_vals[i] = static_cast<uint64_t>(i + 1); // 1..8
    return cfg;
}

// ── Core loop builder ─────────────────────────────────────────────────────────
//
// emit_body(assembler, unroll_iter) is called `cfg.unroll` times, with
// unroll_iter in [0, cfg.unroll). It should emit exactly one instruction
// (or a fixed small group) of the instruction under test.
//
// Registers available to emit_body:
//   x0–x15  : scratch (caller-saved per ABI; we never save/restore them)
//   x19     : loop counter — DO NOT TOUCH
//   x20     : constant source (cfg.source_val) — treat as read-only
//   x21–x28 : callee-saved, not touched by build_loop, available if
//             the generator saves them itself (currently unused)

template<typename F>
static JitPool::TestFn build_loop(const LoopConfig& cfg, F&& emit_body) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // ── Prologue ─────────────────────────────────────────────────────────
    // Allocate 16 bytes and save x19 (loop counter) and x20 (constant source).
    // Stack must remain 16-byte aligned at all times on ARM64.
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));

    // Load loop control and constant-source values.
    a.mov(x19, Imm(cfg.loops));
    a.mov(x20, Imm(cfg.source_val));

    // Seed test registers. Using mov into individual xN is cleaner than
    // trying to vectorise the init; this code runs once at startup.
    for (uint32_t i = 0; i < cfg.num_init_regs && i < 16; ++i)
        a.mov(xr(i), Imm(cfg.init_vals[i]));

    // ── Loop top — aligned to a 64-byte cache line ────────────────────────
    // Alignment prevents the loop from spanning two fetch groups (cache
    // lines). An unaligned loop can show artificially inflated or variable
    // fetch/decode latency that masks the instruction latency we're measuring.
    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // ── Test body (unrolled) ──────────────────────────────────────────────
    for (uint32_t u = 0; u < cfg.unroll; ++u)
        emit_body(a, u);

    // ── Loop control ──────────────────────────────────────────────────────
    a.sub(x19, x19, Imm(1));   // decrement (does NOT write NZCV)
    a.cbnz(x19, loop_top);     // branch if counter != 0

    // ── Epilogue ──────────────────────────────────────────────────────────
    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn)
        fprintf(stderr, "build_loop: AsmJit compile failed\n");
    return fn;
}

// ── Reference function for Tier 2 ratio normalization ────────────────────────

TestFn create_add_latency_ref(uint64_t loops, uint32_t unroll) {
    // Serial ADD chain: x0 = x0 + x20, repeated (unroll) times per iteration.
    // Each ADD depends on the previous result, forcing strictly serial execution.
    // Expected latency: 1 cycle/instruction on all modern ARM64 microarchitectures,
    // so ratio = test_ns_per_insn / ref_ns_per_insn directly gives CPI of the test.
    LoopConfig cfg = default_cfg(loops, unroll);
    return build_loop(cfg, [](a64::Assembler& a, uint32_t) {
        a.add(x0, x0, x20);   // x0 ← x0 + x20 (chained on x0; x20 is constant)
    });
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

// Build a BenchmarkParams for a function that runs (loops) outer iterations
// with (unroll) instructions per iteration.
static BenchmarkParams make_params(const BenchmarkParams& base,
                                   uint64_t loops, uint32_t unroll) {
    BenchmarkParams p         = base;
    p.loops                   = loops;
    p.instructions_per_loop   = unroll;
    return p;
}

// Compile, benchmark once, and immediately release the JIT function.
// Releasing promptly keeps memory pressure low during long runs.
static void run_one(const char* name, JitPool::TestFn fn,
                    const BenchmarkParams& params) {
    if (!fn) return;
    benchmark(fn, name, params);
    g_jit_pool->release(fn);
}

// ════════════════════════════════════════════════════════════════════════════
// Section 1: ADD
// ════════════════════════════════════════════════════════════════════════════
//
// ADD is the canonical integer benchmark. Its latency is 1 cycle on all
// modern ARM64 microarchitectures, so the latency test is primarily a
// sanity check and calibration anchor. The throughput tests are more
// interesting: they reveal how many integer ALU ports handle ADD.
//
// Apple M1–M4:           4–6 integer ALUs, all capable of ADD
// Cortex-A76/A78/X1:     4 integer ALUs
// Snapdragon Oryon:      6 integer ALUs (estimated)
// Cortex-A55 (E-core):   2 integer ALUs

static void run_add_tests(const BenchmarkParams& base,
                          uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── ADD x64 latency ───────────────────────────────────────────────────
    // ADD x0, x0, x1: x0 depends on the previous x0. Each iteration must
    // wait for the previous result. Expect 1 cycle/insn on any modern core.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.add(x0, x0, x1);   // x0 = x0 + x1 (chained on x0)
        });
        snprintf(name, sizeof(name), "ADD x64 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── ADD x64 throughput: sweep chain count ─────────────────────────────
    // Each chain accumulates into a distinct register (x0, x1, x2, ...).
    // Chains are entirely independent: the CPU's out-of-order engine can
    // issue to multiple integer ALU ports simultaneously.
    //
    // UNROLL ROUNDING: the unroll count is rounded DOWN to the nearest
    // multiple of nc so that every chain gets exactly the same number of
    // instructions. Without this, e.g. nc=6 and unroll=32 gives x0 and x1
    // one extra instruction each (32 = 5×6 + 2), making them the critical
    // path and slightly inflating the measured time.
    //
    // MAX CHAINS: x0–x15 are our scratch registers (16 total). x20 holds
    // the constant addend. We stop at 16 chains.
    //
    // SATURATION DETECTION: when adding more chains produces no further
    // improvement in clk/insn, all integer ALU ports are saturated. The
    // number of chains at that point is a lower bound on the ALU port count.
    // (It's a lower bound because the OOO window, register file read bandwidth,
    // or issue queue capacity might limit us before the ALUs do.)
    //
    // x20 is used as a shared constant addend for all chains. This means the
    // register file must supply x20 as a read operand on every instruction,
    // which is a read-bandwidth stress. On most implementations the register
    // file has enough read ports for this not to be the bottleneck.
    {
        static const uint32_t kChainCounts[] = { 2, 3, 4, 6, 8, 10, 12, 16 };
        for (uint32_t nc : kChainCounts) {
            // Round unroll down to nearest multiple of nc, minimum nc itself
            // (at least 1 instruction per chain per iteration).
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            // Initialise all nc chain registers (default_cfg only inits 8).
            cfg.num_init_regs = nc;
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = static_cast<uint64_t>(i + 1); // 1..nc

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.add(xr(u % nc), xr(u % nc), x20);
            });
            snprintf(name, sizeof(name),
                     "ADD x64 tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }

    // ── ADD x64 imm throughput: sweep chain count ────────────────────────
    // Same sweep as the register-form test above, but uses ADD xN, xN, #1
    // (immediate form) instead of ADD xN, xN, x20.
    //
    // The critical difference: the immediate form has exactly ONE register
    // source read per instruction (just the destination-as-source xN). The
    // register-form test above reads xN AND x20 per instruction; when many
    // chains issue simultaneously, all of them must read x20 in the same
    // cycle, potentially stressing the register file's broadcast network.
    //
    // By contrast, this test has no shared register read at all. Each chain
    // reads only its own private register, so the register file read bandwidth
    // is not a bottleneck. This isolates pure ALU port count: the throughput
    // floor here reflects only how many integer ALUs are available.
    //
    // INTERPRETATION: compare with the register-form sweep above.
    //   - If imm-form saturates at a higher chain count → shared x20 read
    //     was limiting the register-form test before the ALUs were full.
    //   - If both sweeps saturate at the same count → register file broadcast
    //     was NOT the bottleneck; the ALU port count is the shared limit.
    //
    // NOTE: the imm form (UBFM alias) encodes differently but decodes to the
    // same micro-op class on all known ARM64 implementations. The latency
    // test below confirms this; any latency difference there would invalidate
    // using imm-form as an ALU port probe.
    {
        static const uint32_t kImmChainCounts[] = { 2, 3, 4, 6, 8, 10, 12, 16 };
        for (uint32_t nc : kImmChainCounts) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            cfg.num_init_regs = nc;
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = static_cast<uint64_t>(i + 1);

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.add(xr(u % nc), xr(u % nc), Imm(1));
            });
            snprintf(name, sizeof(name),
                     "ADD x64 imm tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }

    // ── ADD x64 self-form throughput: sweep chain count ─────────────────
    // ADD xN, xN, xN — Rd=Rn=Rm. Each instruction left-shifts xN by 1
    // (doubles it), and is the only reader/writer of xN. Unlike the
    // register-form sweep (which reads shared x20) and the imm-form sweep
    // (which has one register read), this instruction has TWO register
    // source reads — but both specify the same register name.
    //
    // The architectural encoding requires the processor to read Rn and Rm
    // separately, but since they name the same physical register, a
    // well-designed register file can supply both values from a single
    // read port via internal broadcast. The question is whether the
    // implementation actually does this.
    //
    // THREE POSSIBLE OUTCOMES vs. imm-form:
    //   Same saturation floor → same-name double-read costs nothing; the
    //     register file treats Rn=Rm as a single read. Self-form and
    //     immediate form are equivalent probes of ALU port count.
    //   Slightly higher floor → same-name double-read has a small cost,
    //     perhaps consuming an extra read port per cycle even when Rn=Rm.
    //   Substantially higher floor → the implementation does not
    //     special-case Rn=Rm and pays full two-read bandwidth cost,
    //     similar to the shared-x20 register-form sweep.
    //
    // This disambiguates whether the imm-form sweep is truly representative
    // of "clean" single-source throughput, or whether the instruction
    // encoding itself matters.
    {
        static const uint32_t kSelfChainCounts[] = { 2, 3, 4, 6, 8, 10, 12, 16 };
        for (uint32_t nc : kSelfChainCounts) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            cfg.num_init_regs = nc;
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = static_cast<uint64_t>(i + 1);

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                const auto& r = xr(u % nc);
                a.add(r, r, r);    // xN = xN + xN  (Rd = Rn = Rm)
            });
            snprintf(name, sizeof(name),
                     "ADD x64 self tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }

    // ── ADD w32 latency ───────────────────────────────────────────────────
    // On virtually all ARM64 cores, 32-bit ADD uses the same execution unit
    // as 64-bit ADD with identical latency. A measurable difference here
    // would suggest an unusual microarchitectural treatment of W-form
    // instructions (very unlikely, but worth confirming).
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.add(w0, w0, w1);   // 32-bit form
        });
        snprintf(name, sizeof(name), "ADD w32 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── ADD x64 imm latency ───────────────────────────────────────────────
    // ADD Xd, Xn, #imm12 encodes differently from the register form but
    // should decode to the same micro-op on all known implementations.
    // Confirms that the immediate encoding doesn't add front-end latency.
    // If this matches the register-form latency, the imm throughput sweep
    // above is a valid ALU port count probe.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.add(x0, x0, Imm(1));
        });
        snprintf(name, sizeof(name), "ADD x64 imm latency  (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: SUB and logical operations
// ════════════════════════════════════════════════════════════════════════════
//
// SUB, AND, ORR, EOR, and NEG are typically handled by the same integer
// ALU ports as ADD. Identical latency/throughput here confirms port
// homogeneity. A difference would suggest dedicated or shared ports with
// different dispatch priorities.

static void run_sub_logical_tests(const BenchmarkParams& base,
                                  uint64_t loops, uint32_t unroll) {
    char name[80];

    struct Op {
        const char* label;
        void (*emit_lat)(a64::Assembler&);    // latency chain body
        void (*emit_tput)(a64::Assembler&, uint32_t nc, uint32_t u); // tput body
    };

    // Latency: always chains through x0.
    // Throughput: 4 chains through x0..x3, adding constant x20.
    const Op ops[] = {
        {
            "SUB x64",
            [](a64::Assembler& a) { a.sub(x0, x0, x1); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.sub(xr(u % nc), xr(u % nc), x20);
            }
        },
        {
            "AND x64",
            // AND with x1 (initialized to 2): result is x0 & 2, which
            // oscillates between 0 and 2 but remains a real dependency chain.
            // We avoid AND with a constant-zero operand (which would kill x0).
            [](a64::Assembler& a) { a.and_(x0, x0, x1); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.and_(xr(u % nc), xr(u % nc), x20);
            }
        },
        {
            "ORR x64",
            [](a64::Assembler& a) { a.orr(x0, x0, x1); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.orr(xr(u % nc), xr(u % nc), x20);
            }
        },
        {
            "EOR x64",
            // EOR with a non-zero value: toggles bits each iteration.
            [](a64::Assembler& a) { a.eor(x0, x0, x1); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.eor(xr(u % nc), xr(u % nc), x20);
            }
        },
        {
            "NEG x64",
            // NEG x0, x0: two's complement negation. Involutory: NEG(NEG(x)) = x.
            // The chain is real despite the oscillation.
            [](a64::Assembler& a) { a.neg(x0, x0); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.neg(xr(u % nc), xr(u % nc));
            }
        },
    };

    for (const auto& op : ops) {
        // Latency
        {
            auto emit_lat = op.emit_lat;
            auto cfg = default_cfg(loops, unroll);
            auto fn  = build_loop(cfg, [emit_lat](a64::Assembler& a, uint32_t) {
                emit_lat(a);
            });
            snprintf(name, sizeof(name), "%s latency       (%ux unroll)",
                     op.label, unroll);
            run_one(name, fn, make_params(base, loops, unroll));
        }

        // Throughput (4 chains)
        {
            constexpr uint32_t kTputChains = 4;
            if (kTputChains > unroll) continue;
            auto emit_tput = op.emit_tput;
            auto cfg = default_cfg(loops, unroll);
            auto fn  = build_loop(cfg, [emit_tput](a64::Assembler& a, uint32_t u) {
                emit_tput(a, kTputChains, u);
            });
            snprintf(name, sizeof(name), "%s tput (4 chains, %ux unroll)",
                     op.label, unroll);
            run_one(name, fn, make_params(base, loops, unroll));
        }
    }

    // ── NEG x64 extended throughput sweep ─────────────────────────────────
    // NEG is unary: it reads exactly one register (no shared operand, no
    // broadcast pressure). This makes it an ideal single-source throughput
    // probe comparable to ADD imm and ADD self-form.
    //
    // Comparing NEG saturation vs ADD imm saturation reveals whether the
    // integer ALU ports that handle NEG are the same ports, a subset, or
    // a superset of those that handle ADD.
    //
    // On M1 (expected): NEG saturates at the same floor as ADD imm (~4–5
    // ops/cycle) since both are dispatched to the main integer ALU cluster.
    // Any difference would indicate a dedicated or restricted NEG pipe.
    //
    // UNROLL ROUNDING applied to keep per-chain instruction counts equal.
    {
        static const uint32_t kNegChains[] = { 6, 8, 10, 12, 16 };
        for (uint32_t nc : kNegChains) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            cfg.num_init_regs = nc;
            // Alternate seeds so adjacent chains start with different values;
            // not required for correctness (NEG is involutory regardless) but
            // avoids the trivially-identical state that might get optimized by
            // a very aggressive microarchitecture.
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = static_cast<uint64_t>(i + 1);

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.neg(xr(u % nc), xr(u % nc));
            });
            snprintf(name, sizeof(name),
                     "NEG x64 tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: Shifts
// ════════════════════════════════════════════════════════════════════════════
//
// ARM64 shift instructions come in two flavours:
//
//   Immediate: LSL/LSR/ASR Xd, Xn, #imm
//     These are aliases for UBFM/SBFM. On most cores they share the
//     integer ALU with bitfield extraction and have 1-cycle latency.
//
//   Register (variable): LSLV/LSRV/ASRV/RORV Xd, Xn, Xm
//     The shift amount comes from a register. May or may not share the
//     same execution unit as the immediate form depending on the core.
//     Some cores (e.g., Cortex-A55) have a single shifter unit that is
//     separate from the main integer ALUs.
//
// Interleaving shifts with independent ADD chains (mix test) is particularly
// useful for revealing whether shifts and adds compete for the same ports.

static void run_shift_tests(const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    char name[80];

    auto run_lat = [&](const char* label, auto body) {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 0x0102030405060708ULL; // non-trivial, non-zero
        auto fn = build_loop(cfg, [body](a64::Assembler& a, uint32_t) {
            body(a);
        });
        snprintf(name, sizeof(name), "%s latency       (%ux unroll)", label, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    };

    auto run_tput = [&](const char* label, uint32_t nc, auto body) {
        if (nc > unroll) return;
        auto cfg = default_cfg(loops, unroll);
        for (uint32_t i = 0; i < nc; ++i)
            cfg.init_vals[i] = 0x0102030405060708ULL ^ (static_cast<uint64_t>(i + 1) << 8);
        auto fn = build_loop(cfg, [nc, body](a64::Assembler& a, uint32_t u) {
            body(a, u % nc);
        });
        snprintf(name, sizeof(name), "%s tput (%u chains, %ux unroll)",
                 label, nc, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    };

    // LSL #1 (immediate): alias for UBFM. 1-cycle latency expected.
    run_lat("LSL #1 x64", [](a64::Assembler& a) { a.lsl(x0, x0, Imm(1)); });
    run_tput("LSL #1 x64", 4, [](a64::Assembler& a, uint32_t r) {
        a.lsl(xr(r), xr(r), Imm(1));
    });

    // LSR #1 (logical shift right, immediate).
    run_lat("LSR #1 x64", [](a64::Assembler& a) { a.lsr(x0, x0, Imm(1)); });

    // ASR #1 (arithmetic shift right, immediate).
    run_lat("ASR #1 x64", [](a64::Assembler& a) { a.asr(x0, x0, Imm(1)); });

    // LSLV (shift left, amount from register).
    // x1 is initialized to 1, so x0 shifts left by 1 each iteration.
    // The chain is x0 → x0<<1 → x0<<1 ... ; x0 reaches 0 quickly but
    // each step is a real dependent shift with register-read latency.
    run_lat("LSLV x64",   [](a64::Assembler& a) { a.lslv(x0, x0, x1); });
    run_tput("LSLV x64",  4, [](a64::Assembler& a, uint32_t r) {
        a.lslv(xr(r), xr(r), x1);  // x1 (=2) is shared shift amount
    });

    // ROR #1 (rotate right, immediate). Involutory with period 64.
    run_lat("ROR #1 x64", [](a64::Assembler& a) { a.ror(x0, x0, Imm(1)); });
}

// ════════════════════════════════════════════════════════════════════════════
// Section 4: Multiply and multiply-accumulate
// ════════════════════════════════════════════════════════════════════════════
//
// MUL latency is typically 3 cycles on all recent ARM64 cores, but the
// NUMBER of multiply execution units varies considerably:
//
//   Apple M1 (Firestorm):   2 multiply units → 2 MUL/cycle throughput
//   Apple M2/M3/M4:         2–3 multiply units (vary by core type)
//   Cortex-A78/X1:          1–2 multiply units
//   Snapdragon Oryon:       2 multiply units (estimated)
//   Cortex-A55 (E-core):    1 multiply unit, partially pipelined
//
// MADD (multiply-accumulate) is the base instruction; MUL is its alias
// with the accumulator set to XZR. We test two critical paths through MADD:
//   - Accumulator chain:  x0 = x1*x2 + x0   (latency = acc input latency)
//   - Multiply chain:     x1 = x1*x2 + 0    (latency = mul input latency)
// On most cores these are equal (3 cycles), but some designs have a
// shorter path through the accumulator that enables FMA-style fusion.

static void run_multiply_tests(const BenchmarkParams& base,
                               uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── MUL x64 latency ───────────────────────────────────────────────────
    // x0 = x0 * x1 (chained on x0). x1 is an odd constant (DEADBEEF...F)
    // so the product is never zero mod 2^64 (odd * odd = odd).
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 3ULL;
        cfg.init_vals[1] = 0xDEADBEEFDEADBEEFULL; // odd, never zero
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.mul(x0, x0, x1);
        });
        snprintf(name, sizeof(name), "MUL x64 latency      (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── MUL x64 throughput sweep ──────────────────────────────────────────
    // Sweep independent chains to find the multiply unit saturation point.
    //
    // Theory: with L=3 cycle latency and R multiply units, the throughput
    // floor is 1/R clk/insn (resource-limited) once N ≥ L×R chains are in
    // flight. Fewer chains are latency-limited at L/N clk/insn.
    //
    //   1 unit:  saturation at N=3 chains → floor 1.0 clk/insn
    //   2 units: saturation at N=6 chains → floor 0.5 clk/insn
    //   3 units: saturation at N=9 chains → floor 0.333 clk/insn
    //
    // Sweeping to 8 chains is sufficient to distinguish 1, 2, or 3 units.
    //
    // UNROLL ROUNDING: applied for the same reason as the ADD sweep — to
    // ensure every chain gets the same number of instructions per iteration.
    //
    // NOTE: x20 is the shared multiplier (odd constant). This puts broadcast
    // pressure on the register file, but with MUL's 3-cycle latency the
    // max issue rate is R ≤ 3/cycle even with infinite units. That rate is
    // well within what register file broadcast networks handle without stall.
    {
        static const uint32_t kMulChains[] = { 2, 3, 4, 6, 8 };
        for (uint32_t nc : kMulChains) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg      = default_cfg(loops, actual_unroll);
            cfg.source_val = 0xDEADBEEFDEADBEEFULL; // odd multiplier in x20
            cfg.num_init_regs = nc;
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = static_cast<uint64_t>(i + 1); // non-zero seeds

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.mul(xr(u % nc), xr(u % nc), x20);
            });
            snprintf(name, sizeof(name),
                     "MUL x64 tput         (%u chains, %ux unroll)",
                     nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }

    // ── MADD: accumulator critical path ───────────────────────────────────
    // x0 = x1*x2 + x0: the addition of the accumulator (x0) is the
    // critical dependency. x1 and x2 are stable constants.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 1ULL;                    // accumulator (chains)
        cfg.init_vals[1] = 3ULL;                    // multiplicand (stable)
        cfg.init_vals[2] = 7ULL;                    // multiplier   (stable)
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.madd(x0, x1, x2, x0);  // x0 = x1*x2 + x0
        });
        snprintf(name, sizeof(name), "MADD x64 acc-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── MADD: multiply critical path ──────────────────────────────────────
    // x1 = x1*x2 + 0: x1 is both a source and result of the multiply.
    // x0 is held at 0 as a constant zero accumulator.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 0ULL;  // accumulator = 0 (constant zero)
        cfg.init_vals[1] = 3ULL;  // multiplicand (chains — result goes back here)
        cfg.init_vals[2] = 7ULL;  // multiplier   (stable)
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.madd(x1, x1, x2, x0);  // x1 = x1*x2 + 0
        });
        snprintf(name, sizeof(name), "MADD x64 mul-chain   (%ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 5: Division
// ════════════════════════════════════════════════════════════════════════════
//
// Division is the most microarchitecturally interesting integer operation:
//
//   1. Latency: varies enormously. Apple M1: ~7 cycles (fast!). Cortex-A78:
//      ~12 cycles. Some Qualcomm cores: up to ~24 cycles for 64-bit SDIV.
//
//   2. Pipelineability: most ARM64 dividers are NOT fully pipelined. A new
//      SDIV cannot begin until the previous one completes (or nearly so).
//      Testing 2 independent chains vs 1 reveals this: if it IS pipelined,
//      2 chains will execute in the same wall time as 1. If NOT, 2 chains
//      will take ~2× the wall time (half the throughput).
//
//   3. SDIV vs UDIV: on most cores these share the same divider unit. If
//      SDIV is measurably slower than UDIV, the core has a sign-extension
//      step before dispatch that adds latency.
//
// LOOP COUNT NOTE: with 20-cycle divide latency at 3GHz = ~6.7ns/insn,
// the standard 6M×32 = 192M instructions would take ~1.3 seconds PER SAMPLE.
// We reduce loop count so each sample runs ~100–150ms.
//
// CYCLING CHAIN: naive chaining (x0 = x0 / x1) causes x0 to converge to
// 0 or 1 within ~log_x1(x0_init) iterations, turning the rest into trivial
// 0÷n operations. Instead we use x0 = x20 / x0 with x20 = k² and x0 = k:
//   49 / 7 = 7 → 49 / 7 = 7 → ... stable at 7 every iteration.
// The chain is genuine: each division must wait for the previous quotient.

static void run_divide_tests(const BenchmarkParams& base,
                             uint64_t loops, uint32_t unroll) {
    char name[80];

    // Reduced loop count for slow instructions.
    // Target: ~100ms per sample at ~10-cycle divide latency on a ~3GHz core.
    // 3GHz / 10 cycles = 0.3 billion divides/sec; 100ms → 30M divides.
    // With unroll=8: 30M / 8 = 3.75M outer iterations.
    const uint64_t div_loops  = (loops > 4'000'000) ? 4'000'000 : loops;
    const uint32_t div_unroll = (unroll > 8) ? 8u : unroll;

    auto run_div = [&](const char* label,
                       void (*body)(a64::Assembler&),
                       uint64_t src_val, const uint64_t init_vals[], uint32_t n_init) {
        LoopConfig cfg{};
        cfg.loops        = div_loops;
        cfg.unroll       = div_unroll;
        cfg.source_val   = src_val;
        cfg.num_init_regs = n_init;
        for (uint32_t i = 0; i < n_init && i < 16; ++i)
            cfg.init_vals[i] = init_vals[i];

        auto fn = build_loop(cfg, [body](a64::Assembler& a, uint32_t) { body(a); });
        snprintf(name, sizeof(name), "%s (%ux unroll)", label, div_unroll);
        run_one(name, fn, make_params(base, div_loops, div_unroll));
    };

    // ── UDIV x64 latency ──────────────────────────────────────────────────
    // x0 = x20 / x0 = 49 / x0, starting at 7. Stable cycle: 49/7 = 7.
    {
        const uint64_t inits[] = { 7 };
        run_div("UDIV x64 latency",
                [](a64::Assembler& a) { a.udiv(x0, x20, x0); },
                49, inits, 1);
    }

    // ── SDIV x64 latency ──────────────────────────────────────────────────
    // Same trick. SDIV and UDIV typically share the divider unit; any
    // latency difference reflects sign-extension pre-processing.
    {
        const uint64_t inits[] = { 7 };
        run_div("SDIV x64 latency",
                [](a64::Assembler& a) { a.sdiv(x0, x20, x0); },
                49, inits, 1);
    }

    // ── UDIV throughput: 2 independent chains ─────────────────────────────
    // x0 = 49/x0 and x1 = 49/x1, both starting at 7. These are entirely
    // independent: the CPU can issue both simultaneously if the divider
    // is pipelined or there are 2 divider units.
    //
    // Expected outcome on most ARM64 cores: clk/insn is roughly double
    // that of the 1-chain case (1 non-pipelined divider, serialized).
    // An Apple M-series surprise: they appear to have partial pipelining.
    {
        LoopConfig cfg{};
        cfg.loops        = div_loops;
        cfg.unroll       = div_unroll;
        cfg.source_val   = 49;
        cfg.num_init_regs = 2;
        cfg.init_vals[0] = 7;
        cfg.init_vals[1] = 7;
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            a.udiv(xr(u % 2), x20, xr(u % 2));
        });
        snprintf(name, sizeof(name), "UDIV x64 tput        (2 chains, %ux unroll)", div_unroll);
        run_one(name, fn, make_params(base, div_loops, div_unroll));
    }

    // ── UDIV throughput: 4 independent chains ─────────────────────────────
    {
        LoopConfig cfg{};
        cfg.loops        = div_loops;
        cfg.unroll       = div_unroll;
        cfg.source_val   = 49;
        cfg.num_init_regs = 4;
        for (uint32_t i = 0; i < 4; ++i) cfg.init_vals[i] = 7;
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            a.udiv(xr(u % 4), x20, xr(u % 4));
        });
        snprintf(name, sizeof(name), "UDIV x64 tput        (4 chains, %ux unroll)", div_unroll);
        run_one(name, fn, make_params(base, div_loops, div_unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 6: Bit manipulation
// ════════════════════════════════════════════════════════════════════════════
//
// CLZ, RBIT, REV and their variants are 1-cycle latency on virtually all
// ARM64 cores, but the interesting question is port assignment:
//   - If they share ports with ADD/SUB, then mixing them with ADD reduces
//     total throughput (the mix test in section 7 measures this).
//   - If they have dedicated ports, they can run concurrently with ALU ops.
//
// These instructions are also involutory (RBIT(RBIT(x)) = x) or have short
// periods (CLZ oscillates), making chained latency tests straightforward.

static void run_bit_tests(const BenchmarkParams& base,
                          uint64_t loops, uint32_t unroll) {
    char name[80];

    // Common non-trivial seed for x0 (avoids degenerate all-zero results).
    const uint64_t kSeed = 0x0102030405060708ULL;

    struct BitOp {
        const char* label;
        void (*body)(a64::Assembler&);
    };

    const BitOp ops[] = {
        // CLZ: count leading zeros. 0..63 clz result, then CLZ(1..63) = 6..0.
        { "CLZ x64 latency", [](a64::Assembler& a) { a.clz(x0, x0); } },

        // RBIT: reverse bits. Involutory: RBIT(RBIT(x)) = x.
        { "RBIT x64 latency", [](a64::Assembler& a) { a.rbit(x0, x0); } },

        // REV: reverse bytes (byte-swap entire 64-bit word). Involutory.
        { "REV x64 latency", [](a64::Assembler& a) { a.rev(x0, x0); } },

        // REV16: reverse bytes within each 16-bit halfword.
        { "REV16 x64 latency", [](a64::Assembler& a) { a.rev16(x0, x0); } },

        // REV32: reverse bytes within each 32-bit word.
        { "REV32 x64 latency", [](a64::Assembler& a) { a.rev32(x0, x0); } },
    };

    for (const auto& op : ops) {
        auto emit = op.body;
        auto cfg  = default_cfg(loops, unroll);
        cfg.init_vals[0] = kSeed;
        auto fn = build_loop(cfg, [emit](a64::Assembler& a, uint32_t) { emit(a); });
        snprintf(name, sizeof(name), "%s      (%ux unroll)", op.label, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CLZ and RBIT extended throughput sweeps ───────────────────────────
    // Both CLZ and RBIT are unary single-source instructions with no shared
    // register reads, making them clean throughput probes.
    //
    // CLZ: result oscillates (CLZ(63)=58, CLZ(58)=59, ...) but the chain
    //   is real — each CLZ must wait for the previous result.
    //
    // RBIT: strictly involutory (RBIT(RBIT(x)) = x), which means the chain
    //   alternates between x and ~bitreverse(x). Still a genuine dependency
    //   chain. RBIT is arguably the purest single-source bit-op probe
    //   because its period-2 behaviour makes the value analysis trivial.
    //
    // KEY QUESTION: do CLZ and RBIT saturate at the same floor as ADD imm
    // (~0.220 clk/insn on M1), confirming they share the main integer ALU
    // ports? Or do they saturate at a different floor, indicating they
    // dispatch to a dedicated or partially-shared bit-manipulation unit?
    //
    // M1 Firestorm is known to have a separate "complex integer" unit in
    // addition to the main ALU cluster. If CLZ/RBIT route there instead of
    // (or in addition to) the main ALUs, the saturation floor will differ.
    {
        static const uint32_t kBitChains[] = { 4, 6, 8, 10, 12, 16 };

        // CLZ sweep
        for (uint32_t nc : kBitChains) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            cfg.num_init_regs = nc;
            // Distinct non-zero seeds so chains start in meaningfully
            // different states; XOR with position to avoid all-same.
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = kSeed ^ (static_cast<uint64_t>(i + 1) << 16);

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.clz(xr(u % nc), xr(u % nc));
            });
            snprintf(name, sizeof(name),
                     "CLZ x64 tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }

        // RBIT sweep
        for (uint32_t nc : kBitChains) {
            const uint32_t actual_unroll =
                (unroll >= nc) ? (unroll / nc) * nc : nc;

            auto cfg = default_cfg(loops, actual_unroll);
            cfg.num_init_regs = nc;
            for (uint32_t i = 0; i < nc; ++i)
                cfg.init_vals[i] = kSeed ^ (static_cast<uint64_t>(i + 1) << 8);

            auto fn = build_loop(cfg, [nc](a64::Assembler& a, uint32_t u) {
                a.rbit(xr(u % nc), xr(u % nc));
            });
            snprintf(name, sizeof(name),
                     "RBIT x64 tput (%2u chains, %ux unroll)", nc, actual_unroll);
            run_one(name, fn, make_params(base, loops, actual_unroll));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 7: Mixed / port pressure diagnostics
// ════════════════════════════════════════════════════════════════════════════
//
// These tests deliberately mix instruction types to probe execution port
// heterogeneity. The key question: can the CPU issue instruction type A
// and type B simultaneously (they're on different ports), or do they
// compete for the same port?
//
// Interpretation:
//   If "ADD+MUL mix" achieves clk/insn ≈ min(add_tput, mul_tput), then
//   ADD and MUL are on separate ports and can run fully concurrently.
//   If the mix is slower than either individually, they share a port.
//
// ADD+DIV is particularly interesting because on cores where the integer
// divider is a separate, non-pipelined unit, ADD instructions can proceed
// while the divider is busy. This reveals whether DIV blocks the dispatch
// queue or runs out-of-band.

static void run_mixed_tests(const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── ADD + MUL interleaved ─────────────────────────────────────────────
    // 4 independent ADD chains (x0..x3) and 2 independent MUL chains (x4..x5).
    // All 6 chains are independent of each other. x20 is used as the addend
    // (for ADD) and the multiplier (for MUL) — this is intentional: x20 is
    // a stable constant that creates no cross-chain dependency.
    if (unroll >= 6) {
        auto cfg = default_cfg(loops, unroll);
        cfg.source_val   = 0xDEADBEEFDEADBEEFULL; // odd, safe for multiply
        cfg.init_vals[4] = 3ULL;
        cfg.init_vals[5] = 5ULL;
        cfg.num_init_regs = 6;

        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            switch (u % 6) {
                case 0: a.add(x0, x0, x20); break;
                case 1: a.add(x1, x1, x20); break;
                case 2: a.mul(x4, x4, x20); break;
                case 3: a.add(x2, x2, x20); break;
                case 4: a.add(x3, x3, x20); break;
                case 5: a.mul(x5, x5, x20); break;
                default: break;
            }
        });
        snprintf(name, sizeof(name),
                 "ADD+MUL mix (4+2 chains, %ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── ADD + DIV interleaved ─────────────────────────────────────────────
    // 2 independent ADD chains and 1 DIV chain, all independent.
    // On cores with a separate divider that runs concurrently with the
    // integer ALUs, the effective throughput approaches the individual
    // throughputs: ADDs proceed unimpeded while DIV churns.
    //
    // Uses the reduced loop count from divide tests to keep runtime sane.
    {
        const uint64_t div_loops  = (loops > 4'000'000) ? 4'000'000 : loops;
        const uint32_t div_unroll = (unroll > 8) ? 8u : unroll;
        if (div_unroll >= 3) {
            LoopConfig cfg{};
            cfg.loops        = div_loops;
            cfg.unroll       = div_unroll;
            cfg.source_val   = 49; // x20 = 49 for div chain; also used as addend
            cfg.num_init_regs = 3;
            cfg.init_vals[0] = 1ULL; // ADD chain 0
            cfg.init_vals[1] = 1ULL; // ADD chain 1
            cfg.init_vals[2] = 7ULL; // DIV chain  (49/7 = 7, cycles)

            auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
                switch (u % 3) {
                    case 0: a.add(x0,  x0,  x20);       break; // x0  += 49
                    case 1: a.add(x1,  x1,  x20);       break; // x1  += 49
                    case 2: a.udiv(x2, x20, x2);         break; // x2   = 49/x2
                    default: break;
                }
            });
            snprintf(name, sizeof(name),
                     "ADD+DIV mix (2+1 chains, %ux unroll)", div_unroll);
            run_one(name, fn, make_params(base, div_loops, div_unroll));
        }
    }

    // ── ADD + CLZ interleaved ─────────────────────────────────────────────
    // 4 ADD chains and 2 CLZ chains, all independent. Since CLZ often
    // shares execution ports with bitfield ops (and possibly with integer
    // ALUs on some cores), this mix reveals port contention more directly
    // than the ADD+MUL mix.
    if (unroll >= 6) {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[4] = 0x0102030405060708ULL;
        cfg.init_vals[5] = 0x0807060504030201ULL;
        cfg.num_init_regs = 6;

        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            switch (u % 6) {
                case 0: a.add(x0, x0, x20); break;
                case 1: a.add(x1, x1, x20); break;
                case 2: a.clz(x4, x4);      break;
                case 3: a.add(x2, x2, x20); break;
                case 4: a.add(x3, x3, x20); break;
                case 5: a.clz(x5, x5);      break;
                default: break;
            }
        });
        snprintf(name, sizeof(name),
                 "ADD+CLZ mix (4+2 chains, %ux unroll)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_integer_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;

    printf("\n── ADD ─────────────────────────────────────────────────────────\n");
    run_add_tests(base_params, loops, unroll);

    printf("\n── SUB / logical ───────────────────────────────────────────────\n");
    run_sub_logical_tests(base_params, loops, unroll);

    printf("\n── Shifts ──────────────────────────────────────────────────────\n");
    run_shift_tests(base_params, loops, unroll);

    printf("\n── Multiply ────────────────────────────────────────────────────\n");
    run_multiply_tests(base_params, loops, unroll);

    printf("\n── Divide ──────────────────────────────────────────────────────\n");
    run_divide_tests(base_params, loops, unroll);

    printf("\n── Bit manipulation ────────────────────────────────────────────\n");
    run_bit_tests(base_params, loops, unroll);

    printf("\n── Mixed / port pressure ───────────────────────────────────────\n");
    run_mixed_tests(base_params, loops, unroll);
}

} // namespace arm64bench::gen
