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
// LOOP STRUCTURE
//   The stack frame, the loop counter, the 64-byte loop alignment and the
//   SUB + CBNZ loop control all come from gen::build_loop() in gen_common.h.
//   The detail that matters for this file: the counter decrement is SUB and
//   not SUBS, so it never writes NZCV. The flag-consuming tests here
//   (ADDS + CSEL, CMP + CCMP) would otherwise pick up a false dependency on
//   the loop counter.
//
// SLOW INSTRUCTION LOOP SCALING
//   Instructions like SDIV/UDIV have latencies of ~10–25 cycles, making
//   them 10–25× slower per iteration than ADD. Running them with the same
//   loop count as ADD would produce 9+ second samples. We scale loops down
//   proportionally so samples stay in the ~100ms range.

#include "gen_integer.h"
#include "gen_common.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

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

// The constant every test loads into x20 unless it asks for another one.
inline constexpr uint64_t kDefaultSourceVal = 0x12345678ULL;

// Sensible defaults: non-zero registers, a useful constant in x20.
static LoopConfig default_cfg(uint64_t loops, uint32_t unroll) {
    LoopConfig cfg{};
    cfg.loops        = loops;
    cfg.unroll       = unroll;
    cfg.source_val   = kDefaultSourceVal;   // non-zero constant for x20
    cfg.num_init_regs = 8;
    for (uint32_t i = 0; i < 8; ++i)
        cfg.init_vals[i] = static_cast<uint64_t>(i + 1); // 1..8
    return cfg;
}

// ── Core loop builder ─────────────────────────────────────────────────────────
//
// A LoopConfig is a register-seeding recipe and nothing more: gen::build_loop()
// supplies the frame, the counter and the loop control, and this wrapper turns
// the config into the `setup` pass that runs once before the loop.
//
// emit_body(assembler, unroll_iter) is called `cfg.unroll` times, with
// unroll_iter in [0, cfg.unroll). It should emit exactly one instruction
// (or a fixed small group) of the instruction under test.
//
// Registers available to emit_body:
//   x0–x15  : scratch (caller-saved per ABI; we never save/restore them)
//   x19     : loop counter — DO NOT TOUCH
//   x20     : constant source (cfg.source_val) — treat as read-only
//   x21–x22 : saved and restored by gen::build_loop; free, currently unused
//   x30     : saved and restored by gen::build_loop; bodies may BL

template<typename F>
static JitPool::TestFn build_loop(const LoopConfig& cfg, F&& emit_body) {
    return gen::build_loop(cfg.loops, cfg.unroll,
        [&cfg](a64::Assembler& a) {
            a.mov(x20, Imm(cfg.source_val));

            // Seed test registers. Using mov into individual xN is cleaner
            // than trying to vectorise the init; this code runs once.
            for (uint32_t i = 0; i < cfg.num_init_regs && i < 16; ++i)
                a.mov(xr(i), Imm(cfg.init_vals[i]));
        },
        static_cast<F&&>(emit_body));
}

// The LoopConfig equivalent for a chain_sweep(): the default constant in x20,
// and the nc chain registers seeded 1..nc so no chain starts at zero. Sweeps
// that need other seeds spell them out in their own setup lambda.
static void seed_chains(a64::Assembler& a, uint32_t nc,
                        uint64_t source_val = kDefaultSourceVal) {
    a.mov(x20, Imm(source_val));
    for (uint32_t i = 0; i < nc; ++i)
        a.mov(xr(i), Imm(static_cast<uint64_t>(i + 1)));
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
        run_one(name, fn, params_for(base, loops, unroll));
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
    chain_sweep(base, loops, unroll, "ADD x64 tput", kWideChains,
        [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.add(xr(u % nc), xr(u % nc), x20);
        });

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
    chain_sweep(base, loops, unroll, "ADD x64 imm tput", kWideChains,
        [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.add(xr(u % nc), xr(u % nc), Imm(1));
        });

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
    chain_sweep(base, loops, unroll, "ADD x64 self tput", kWideChains,
        [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            const auto& r = xr(u % nc);
            a.add(r, r, r);    // xN = xN + xN  (Rd = Rn = Rm)
        });

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
        run_one(name, fn, params_for(base, loops, unroll));
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
        run_one(name, fn, params_for(base, loops, unroll));
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
            snprintf(name, sizeof(name), "%s latency        (%ux unroll)",
                     op.label, unroll);
            run_one(name, fn, params_for(base, loops, unroll));
        }

        // Throughput (4 chains)
        {
            static const uint32_t kTputChains[] = { 4 };
            if (kTputChains[0] > unroll) continue;
            auto emit_tput = op.emit_tput;
            snprintf(name, sizeof(name), "%s tput", op.label);
            chain_sweep(base, loops, unroll, name, kTputChains,
                [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
                [emit_tput](a64::Assembler& a, uint32_t nc, uint32_t u) {
                    emit_tput(a, nc, u);
                });
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
        // Distinct seeds so adjacent chains start with different values; not
        // required for correctness (NEG is involutory regardless) but avoids
        // the trivially-identical state that might get optimized away by a
        // very aggressive microarchitecture.
        chain_sweep(base, loops, unroll, "NEG x64 tput", kNegChains,
            [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.neg(xr(u % nc), xr(u % nc));
            });
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
        snprintf(name, sizeof(name), "%s latency        (%ux unroll)", label, unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    };

    auto run_tput = [&](const char* label, uint32_t nc, auto body) {
        if (nc > unroll) return;
        const uint32_t chains[] = { nc };
        snprintf(name, sizeof(name), "%s tput", label);
        chain_sweep(base, loops, unroll, name, chains,
            [](a64::Assembler& a, uint32_t n) {
                a.mov(x20, Imm(kDefaultSourceVal));
                for (uint32_t i = 0; i < n; ++i)
                    a.mov(xr(i), Imm(0x0102030405060708ULL ^
                                     (static_cast<uint64_t>(i + 1) << 8)));
            },
            [body](a64::Assembler& a, uint32_t n, uint32_t u) { body(a, u % n); });
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
        run_one(name, fn, params_for(base, loops, unroll));
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
    // x20 = odd multiplier; chains seeded 1..nc so no product starts at zero.
    chain_sweep(base, loops, unroll, "MUL x64 tput", kDefaultChains,
        [](a64::Assembler& a, uint32_t nc) {
            seed_chains(a, nc, 0xDEADBEEFDEADBEEFULL);
        },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.mul(xr(u % nc), xr(u % nc), x20);
        });

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
        run_one(name, fn, params_for(base, loops, unroll));
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
        run_one(name, fn, params_for(base, loops, unroll));
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
        run_one(name, fn, params_for(base, div_loops, div_unroll));
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

    // ── UDIV throughput: 2 and 4 independent chains ───────────────────────
    // Every chain runs xN = 49/xN starting at 7, so all of them are entirely
    // independent: the CPU can issue them simultaneously if the divider is
    // pipelined or there is more than one divider unit.
    //
    // Expected outcome on most ARM64 cores: clk/insn is roughly double that
    // of the 1-chain case (1 non-pipelined divider, serialized).
    // An Apple M-series surprise: they appear to have partial pipelining.
    {
        chain_sweep(base, div_loops, div_unroll, "UDIV x64 tput", { 2, 4 },
            [](a64::Assembler& a, uint32_t nc) {
                a.mov(x20, Imm(49));
                for (uint32_t i = 0; i < nc; ++i) a.mov(xr(i), Imm(7));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.udiv(xr(u % nc), x20, xr(u % nc));
            });
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
    static constexpr uint64_t kSeed = 0x0102030405060708ULL;

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
        snprintf(name, sizeof(name), "%-18s   (%ux unroll)", op.label, unroll);
        run_one(name, fn, params_for(base, loops, unroll));
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

        // Distinct non-zero seeds so chains start in meaningfully different
        // states; the XOR with the chain position avoids all-same.
        chain_sweep(base, loops, unroll, "CLZ x64 tput", kBitChains,
            [](a64::Assembler& a, uint32_t nc) {
                a.mov(x20, Imm(kDefaultSourceVal));
                for (uint32_t i = 0; i < nc; ++i)
                    a.mov(xr(i), Imm(kSeed ^ (static_cast<uint64_t>(i + 1) << 16)));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.clz(xr(u % nc), xr(u % nc));
            });
        chain_sweep(base, loops, unroll, "RBIT x64 tput", kBitChains,
            [](a64::Assembler& a, uint32_t nc) {
                a.mov(x20, Imm(kDefaultSourceVal));
                for (uint32_t i = 0; i < nc; ++i)
                    a.mov(xr(i), Imm(kSeed ^ (static_cast<uint64_t>(i + 1) << 8)));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.rbit(xr(u % nc), xr(u % nc));
            });
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
        run_one(name, fn, params_for(base, loops, unroll));
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
            run_one(name, fn, params_for(base, div_loops, div_unroll));
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
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 8: Integer instruction gaps
// ════════════════════════════════════════════════════════════════════════════
//
// Instructions that are present in the existing coverage but deserve more
// thorough throughput sweeps, plus new instructions not yet tested.
//
// EXTR (extract register):
//   EXTR Xd, Xn, Xm, #lsb  — Xd = (Xn:Xm)[lsb+63:lsb]
//   Equivalent to a 64-bit right-rotation of the 128-bit concatenation (Xn:Xm).
//   Used heavily in cryptography (GCM, ChaCha20) and arbitrary-precision
//   arithmetic. Architecturally, EXTR with Xn=Xm is ROR.
//   Latency expected: 1 cycle (shares the shifter/bitfield unit with LSL/LSR).
//
// UMULH (unsigned multiply high):
//   UMULH Xd, Xn, Xm  — Xd = (Xn * Xm) >> 64  (upper 64 bits of 128-bit product)
//   Used for: 128-bit multiply, Montgomery modular reduction (crypto), hash
//   functions (MurmurHash3, xxHash), compiler division-by-constant elimination.
//   Latency: same as MUL on virtually all implementations (3 cycles), since
//   both use the same 128-bit multiplier hardware. The throughput reveals
//   whether the multiplier can sustain one result per cycle for this form.

static void run_integer_gap_tests(const BenchmarkParams& base,
                                   uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── EXTR latency ──────────────────────────────────────────────────────
    // EXTR x0, x1, x0, #32: x0 = upper-32-of-x0 | lower-32-of-x1
    // Chains through x0; x1 is a stable constant (= 2 from default_cfg).
    // Expected: 1 cycle latency on all modern ARM64 cores.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.extr(x0, x1, x0, Imm(32));
        });
        snprintf(name, sizeof(name), "EXTR x64 latency       (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── EXTR throughput sweep ─────────────────────────────────────────────
    // Each chain: EXTR xN, x_const, xN, #32 — all read x1 as the upper source.
    // This stresses the register-file broadcast for x1 (shared constant read)
    // similar to the ADD x64 throughput sweep.
    {
        chain_sweep(base, loops, unroll, "EXTR x64 tput", { 4, 6, 8, 10, 12, 16 },
            [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.extr(xr(u % nc), x1, xr(u % nc), Imm(32));
            });
    }

    // ── UMULH latency ─────────────────────────────────────────────────────
    // UMULH x0, x0, x20: x0 = upper-64(x0 × x20)
    // Chain through x0; x20 = odd constant from default_cfg.source_val.
    //
    // Note: with small initial x0 values, UMULH often produces 0 for the
    // first few iterations (since x0 < 2^32 and x20 < 2^32 → product < 2^64).
    // This does NOT affect latency measurement: the CPU must still wait for
    // the previous UMULH to produce its result before starting the next one.
    // Zero is not a special-cased input for the multiplier on any known core.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.source_val   = 0xDEADBEEFDEADBEEFULL;   // large odd constant
        cfg.init_vals[0] = 0xCAFEBABECAFEBABEULL;   // large initial value
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.umulh(x0, x0, x20);
        });
        snprintf(name, sizeof(name), "UMULH x64 latency      (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── UMULH throughput sweep ────────────────────────────────────────────
    // Reveal the number of multiplier units that support UMULH.
    // On all known ARM64 implementations, UMULH uses the same 128-bit
    // multiplier as MUL, so UMULH saturation should match MUL saturation.
    chain_sweep(base, loops, unroll, "UMULH x64 tput", kDefaultChains,
        [](a64::Assembler& a, uint32_t nc) {
            a.mov(x20, Imm(0xDEADBEEFDEADBEEFULL));
            for (uint32_t i = 0; i < nc; ++i)
                a.mov(xr(i), Imm(0xCAFEBABECAFEBABEULL ^ static_cast<uint64_t>(i + 1)));
        },
        [](a64::Assembler& a, uint32_t nc, uint32_t u) {
            a.umulh(xr(u % nc), xr(u % nc), x20);
        });
}

// ════════════════════════════════════════════════════════════════════════════
// Section 9: Conditional select (CSEL / CSINV / CSNEG)
// ════════════════════════════════════════════════════════════════════════════
//
// CSEL Xd, Xn, Xm, cond  → Xd = (cond) ? Xn : Xm
// CSINV Xd, Xn, Xm, cond → Xd = (cond) ? Xn : ~Xm
// CSNEG Xd, Xn, Xm, cond → Xd = (cond) ? Xn : -Xm
//
// These instructions read the NZCV condition flags register. Measuring
// their latency is tricky because:
//   1. The loop counter uses SUB + CBNZ (never writes NZCV) — good.
//   2. We need a flag-writing instruction to feed each CSEL.
//
// LATENCY CHAIN DESIGN
//   Interleave ADDS x0, x0, x20 with CSEL x0, x0, x1, NE in alternating
//   unroll slots. The chain is:
//
//     ADDS: x0_new = x0 + x20, NZCV_new = flags(x0 + x20)   [x20 ≠ 0 → NE always]
//     CSEL: x0_out = (NE) ? x0_new : x1                      [always selects x0_new]
//
//   Each CSEL depends on both x0_new (data dep) and NZCV_new (flag dep) from
//   the preceding ADDS. The ADDS in the next slot depends on x0_out from CSEL.
//   This forms a strict serial chain through (ADDS + CSEL) pairs.
//
//   Reported clk/insn = (ADDS_lat + CSEL_lat) / 2.
//   Since ADDS_lat = 1 cycle on all modern ARM64 cores:
//     clk/insn ≈ 1.0 means CSEL_lat = 1 cycle.
//     clk/insn ≈ 1.5 means CSEL_lat = 2 cycles.
//
// THROUGHPUT DESIGN
//   N independent (ADDS+CSEL) pair chains across N registers.
//   Interleaving pattern: N ADDS slots followed by N CSEL slots.
//   OOO execution with flag renaming allows all N ADDS to issue in
//   parallel, and all N CSELs to issue in parallel once the ADDs complete.
//
//   Throughput floor = max(1/n_adds_units, 1/n_csel_units) cycles per insn.
//   Saturation chain count reveals the bottleneck unit count.

static void run_csel_tests(const BenchmarkParams& base,
                            uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── ADDS + CSEL latency chain ──────────────────────────────────────────
    // Alternating ADDS (even slots) and CSEL (odd slots) in the unrolled body.
    // x20 = 0x12345678 (non-zero constant from default_cfg → NE always set).
    // x1  = 2 (false arm of CSEL, never selected since NE=true).
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 1ULL;   // x0: chain accumulator
        cfg.init_vals[1] = 2ULL;   // x1: false arm (never selected)
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            if (u % 2 == 0)
                a.adds(x0, x0, x20);                     // flags: NE=true (x20 ≠ 0)
            else
                a.csel(x0, x0, x1, CondCode::kNE);       // always selects x0 (true arm)
        });
        snprintf(name, sizeof(name), "ADDS+CSEL chain      (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── ADDS + CSINV latency chain ─────────────────────────────────────────
    // CSINV: true arm (NE) selects Xn = x0 unchanged.
    // False arm would produce ~x1 = ~2, but is never taken.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 1ULL;
        cfg.init_vals[1] = 2ULL;
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            if (u % 2 == 0)
                a.adds(x0, x0, x20);
            else
                a.csinv(x0, x0, x1, CondCode::kNE);      // always selects x0
        });
        snprintf(name, sizeof(name), "ADDS+CSINV chain     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── ADDS + CSNEG latency chain ─────────────────────────────────────────
    // CSNEG: true arm (NE) selects Xn = x0 unchanged.
    // False arm would produce -(x1) = -2, but is never taken.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 1ULL;
        cfg.init_vals[1] = 2ULL;
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            if (u % 2 == 0)
                a.adds(x0, x0, x20);
            else
                a.csneg(x0, x0, x1, CondCode::kNE);      // always selects x0
        });
        snprintf(name, sizeof(name), "ADDS+CSNEG chain     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CSEL throughput sweep ─────────────────────────────────────────────
    // nc independent ADDS+CSEL pair chains. Interleaved pattern:
    //   slots 0..nc-1    : ADDS x[i], x[i], x20  (all nc chains)
    //   slots nc..2nc-1  : CSEL x[i], x[i], xzr, ne  (all nc chains)
    //
    // OOO flag renaming ensures each CSEL reads its own chain's NZCV,
    // keeping chains fully independent.
    //
    // xzr (always zero) is the false arm — never selected since NE=true.
    //
    // Interpretation: saturation chain count ≈ min(n_adds_units, n_csel_units).
    // Each pair is 2 instructions; clk/insn at saturation ≈ 1 / min_units.
    {
        static const uint32_t kCselChains[] = { 2, 4, 6, 8 };
        // group = 2: the unroll is rounded to a multiple of 2*nc so that every
        // iteration holds a whole number of nc-ADDS / nc-CSEL phases.
        chain_sweep(base, loops, unroll, "CSEL tput", kCselChains,
            [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                const uint32_t phase = u % (2 * nc);
                const uint32_t chain = phase % nc;
                if (phase < nc)
                    a.adds(xr(chain), xr(chain), x20);
                else
                    a.csel(xr(chain), xr(chain), xzr, CondCode::kNE);
            },
            /*group=*/2);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 10: Bitfield insert/extract (BFI, BFXIL, UBFX, SBFX)
// ════════════════════════════════════════════════════════════════════════════
//
// Why these matter: BFI is the canonical instruction for x86 emulators
// (Prism on Windows ARM64, Rosetta on Apple Silicon) when:
//   - merging an 8-bit or 16-bit subregister write (AL, AX) into the host
//     64-bit GPR holding the emulated x64 register;
//   - splicing live RIP bits into a hash-table base pointer for a fast
//     translation-cache lookup (Prism uses BFI of x9 lower-13 bits into
//     x15 shifted left by 4 — i.e. bits [16:4] of x15).
//
// Pre-Cortex-X1 Snapdragon implementations had BFI latency = 2 cycles, while
// Apple Silicon has always done BFI in 1 cycle. That difference shows up in
// every emulator hot path. This section measures it directly.
//
// All instructions in this section are baseline ARMv8.0-A; no runtime
// feature detection is required. The tests are JIT-emitted via AsmJit and
// therefore do not depend on any compile-time __ARM_FEATURE_* macros.
//
// BFI / BFXIL DESTINATION DEPENDENCY
//   Both BFI and BFXIL inherently *read* their destination register to
//   preserve the non-inserted bits, so a chain that feeds Xd back into
//   itself is a true latency chain — the next BFI cannot retire until
//   the previous one's destination is available. This is the ARM64 analogue
//   of the x86 POPCNT false-dest-dependency famous on Intel pre-Ice Lake,
//   except here the dependency is inherent to the instruction's semantics.
//
//   See run_bfi_dependency_tests() in gen_pitfalls.cpp for the explicit
//   stress-test variant that defeats any µarch attempt to break the dep.

static void run_bitfield_tests_impl(const BenchmarkParams& base,
                                    uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── BFI low-bits latency (8-bit subregister-merge idiom) ──────────────
    // BFI x0, x1, #0, #8: insert lower 8 bits of x1 into x0[7:0].
    // Models "MOV AL, ..." style emulator merges. Chains via x0 → x0.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.bfi(x0, x1, Imm(0), Imm(8));
        });
        snprintf(name, sizeof(name), "BFI x64 lat (#0,#8)    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── BFI mid-bits latency (Prism hash-table-lookup idiom) ──────────────
    // BFI x0, x1, #4, #13: insert 13 bits of x1 into x0[16:4].
    // This is exactly the pattern Prism uses to splice RIP bits into a
    // pointer in a hash-table lookup (per Darek Mihocka).
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.bfi(x0, x1, Imm(4), Imm(13));
        });
        snprintf(name, sizeof(name), "BFI x64 lat (#4,#13)   (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── BFXIL latency (16-bit subregister-merge idiom) ────────────────────
    // BFXIL x0, x1, #0, #16: insert lower 16 bits of x1 into x0[15:0].
    // Models "MOV AX, ..." style emulator merges. Like BFI, BFXIL reads Xd
    // to preserve the non-inserted bits, so chaining x0 → x0 is a real
    // latency chain.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.bfxil(x0, x1, Imm(0), Imm(16));
        });
        snprintf(name, sizeof(name), "BFXIL x64 lat (#0,#16) (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── UBFX latency (pure extract, no read-modify) ───────────────────────
    // UBFX x0, x0, #4, #13: x0 = zero-extended bits[16:4] of x0.
    // Unlike BFI/BFXIL, UBFX *fully* writes Xd — no read of the previous
    // value. The chain dependency here comes from x0 being both source and
    // destination. Latency is therefore "true" UBFX latency, but the µarch
    // is free to break the dep on Xd as a partial-write rename — most
    // implementations don't bother since UBFX is always a full-width write.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.ubfx(x0, x0, Imm(4), Imm(13));
        });
        snprintf(name, sizeof(name), "UBFX x64 lat (#4,#13)  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── SBFX latency (signed extract) ─────────────────────────────────────
    // Same shape as UBFX but sign-extends. On all known ARM64 cores SBFX
    // and UBFX share the same shifter pipeline and have identical latency.
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.sbfx(x0, x0, Imm(4), Imm(13));
        });
        snprintf(name, sizeof(name), "SBFX x64 lat (#4,#13)  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── BFI throughput sweep ──────────────────────────────────────────────
    // N independent BFI chains, each merging x1 into its own xN destination.
    // x1 is read by all chains (broadcast); each chain's xN is the chained
    // accumulator. Saturation chain count = number of execution units
    // capable of issuing BFI per cycle.
    {
        chain_sweep(base, loops, unroll, "BFI x64 tput", { 2, 4, 6, 8 },
            // One register past the chain set (x_nc) is the shared BFI source.
            [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc + 1); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.bfi(xr(u % nc), xr(nc), Imm(4), Imm(13));
            });
    }
}

void run_bitfield_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;
    section("Bitfield insert/extract");
    run_bitfield_tests_impl(base_params, loops, unroll);
}

// ════════════════════════════════════════════════════════════════════════════
// Section 11: Misc bit-op gaps (CLS, BIC, ORN, EON, CCMP)
// ════════════════════════════════════════════════════════════════════════════
//
// CLS counts leading sign bits (the CLZ counterpart for signed values).
// BIC/ORN/EON are AND/ORR/EOR with the second operand inverted — they
// are useful for masking and complement-arithmetic patterns and may share
// pipelines with their plain counterparts.
// CCMP is the flag-producing companion to CSEL: it computes a comparison
// only if a previous condition holds, allowing branchless && / || chains.
//
// All baseline ARMv8.0-A; no runtime detection needed.

static void run_misc_bitops_tests_impl(const BenchmarkParams& base,
                                       uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── CLS latency ───────────────────────────────────────────────────────
    // CLS x0, x0: count leading sign bits of x0 (number of bits matching
    // bit 63 minus 1). Chain via x0 → x0. On all known ARM64 cores CLS
    // shares the priority encoder with CLZ and has identical latency.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 0xFFFFFFFFFFFFFFFEULL;   // produces a useful CLS result
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.cls(x0, x0);
        });
        snprintf(name, sizeof(name), "CLS x64 latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CLS throughput sweep ──────────────────────────────────────────────
    {
        chain_sweep(base, loops, unroll, "CLS x64 tput", { 2, 4, 6, 8 },
            // Negated seeds: CLS on a mostly-ones value returns a useful count.
            [](a64::Assembler& a, uint32_t nc) {
                a.mov(x20, Imm(kDefaultSourceVal));
                for (uint32_t i = 0; i < nc; ++i)
                    a.mov(xr(i), Imm(~static_cast<uint64_t>(i + 1)));
            },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.cls(xr(u % nc), xr(u % nc));
            });
    }

    // ── BIC latency (x0 ← x0 AND ~x20) ────────────────────────────────────
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.bic(x0, x0, x20);
        });
        snprintf(name, sizeof(name), "BIC x64 latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── BIC throughput sweep ──────────────────────────────────────────────
    {
        chain_sweep(base, loops, unroll, "BIC x64 tput", { 2, 4, 6, 8 },
            [](a64::Assembler& a, uint32_t nc) { seed_chains(a, nc); },
            [](a64::Assembler& a, uint32_t nc, uint32_t u) {
                a.bic(xr(u % nc), xr(u % nc), x20);
            });
    }

    // ── ORN latency (x0 ← x0 OR ~x20) ─────────────────────────────────────
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.orn(x0, x0, x20);
        });
        snprintf(name, sizeof(name), "ORN x64 latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── EON latency (x0 ← x0 XOR ~x20) ────────────────────────────────────
    {
        auto cfg = default_cfg(loops, unroll);
        auto fn  = build_loop(cfg, [](a64::Assembler& a, uint32_t) {
            a.eon(x0, x0, x20);
        });
        snprintf(name, sizeof(name), "EON x64 latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CCMP flag chain ───────────────────────────────────────────────────
    // CCMP Xn, Xm, #nzcv, cond:
    //   if (cond) flags = compare(Xn, Xm)
    //   else      flags = #nzcv
    //
    // To form a serial flag-dependent chain we need each CCMP to depend on
    // the flags produced by the previous one. Every CCMP after the first
    // uses the *previous* flags as its predicate (cond=NE), and each writes
    // a new NZCV. Since x0 == x20 is false in our setup (x0 starts at 1,
    // x20 = 0x12345678), the comparison sets ZF=0 → NE = true → next CCMP
    // re-takes the comparison branch. Chain stays alive forever.
    //
    // Reported clk/insn ≈ CMP latency (1) for the first, then CCMP latency
    // for each subsequent. With many unrolled CCMPs the average converges
    // on CCMP latency.
    {
        auto cfg = default_cfg(loops, unroll);
        cfg.init_vals[0] = 1ULL;
        auto fn = build_loop(cfg, [](a64::Assembler& a, uint32_t u) {
            if (u == 0)
                a.cmp(x0, x20);  // seed flags (NE will be true: 1 != 0x12345678)
            else
                a.ccmp(x0, x20, Imm(0), CondCode::kNE);
        });
        snprintf(name, sizeof(name), "CMP+CCMP flag chain    (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

void run_misc_bitops_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;
    section("Misc bit-ops (CLS / BIC / ORN / EON / CCMP)");
    run_misc_bitops_tests_impl(base_params, loops, unroll);
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_integer_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;

    section("ADD");
    run_add_tests(base_params, loops, unroll);

    section("SUB / logical");
    run_sub_logical_tests(base_params, loops, unroll);

    section("Shifts");
    run_shift_tests(base_params, loops, unroll);

    section("Multiply");
    run_multiply_tests(base_params, loops, unroll);

    section("Divide");
    run_divide_tests(base_params, loops, unroll);

    section("Bit manipulation");
    run_bit_tests(base_params, loops, unroll);

    section("Mixed / port pressure");
    run_mixed_tests(base_params, loops, unroll);

    section("Integer gaps");
    run_integer_gap_tests(base_params, loops, unroll);

    section("Conditional select");
    run_csel_tests(base_params, loops, unroll);

    run_bitfield_tests(base_params);
    run_misc_bitops_tests(base_params);
}

} // namespace arm64bench::gen
