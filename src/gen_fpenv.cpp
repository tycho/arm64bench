// gen_fpenv.cpp
// FPCR / FPSR access cost tests. See gen_fpenv.h.
//
// ── Register conventions ─────────────────────────────────────────────────────
//
// On top of build_loop's (x19 = counter):
//   x20  FPCR as the function found it ("host" value). Loaded by
//        build_fpenv_loop before the generator's setup, written back after
//        the loop. Bodies may write it to FPCR but must not change it.
//   x21  the "guest" FPCR value of the test, x20 with some bits flipped
//   x22  free
//
// ── Loop counts ───────────────────────────────────────────────────────────────
//
// A serialising system-register write can cost tens of cycles, so the rows
// that are nothing but MRS/MSR run far fewer instructions per call than the
// ALU tests do: 16 M is 4 ms at 1 clk on a 4 GHz core and 0.4 s at 100 clk.

#include "gen_fpenv.h"
#include "gen_common.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── FPCR fields ───────────────────────────────────────────────────────────────

static constexpr uint64_t kFpcrFIZ   = 1ull << 0;    // FEAT_AFP: flush denormal inputs to zero
static constexpr uint64_t kFpcrAH    = 1ull << 1;    // FEAT_AFP: alternate (x86-style) handling
static constexpr uint64_t kFpcrNEP   = 1ull << 2;    // FEAT_AFP: scalar ops merge into the destination
static constexpr uint64_t kFpcrRZ    = 3ull << 22;   // RMode = 0b11, round toward zero
static constexpr uint64_t kFpcrFZ    = 1ull << 24;   // flush-to-zero

static inline void mrs_fpcr(a64::Assembler& a, const Gp& r) { a.mrs(r, Imm(Predicate::SysReg::kFPCR)); }
static inline void msr_fpcr(a64::Assembler& a, const Gp& r) { a.msr(Imm(Predicate::SysReg::kFPCR), r); }
static inline void mrs_fpsr(a64::Assembler& a, const Gp& r) { a.mrs(r, Imm(Predicate::SysReg::kFPSR)); }
static inline void msr_fpsr(a64::Assembler& a, const Gp& r) { a.msr(Imm(Predicate::SysReg::kFPSR), r); }

static constexpr uint64_t kFpsrIXC = 1ull << 4;      // cumulative inexact

// ── Loop builder ──────────────────────────────────────────────────────────────

// build_loop with x20 = FPCR at entry and FPCR restored from x20 at exit.
template<class FSetup, class FBody>
static JitPool::TestFn build_fpenv_loop(uint64_t loops, uint32_t unroll,
                                        FSetup&& setup, FBody&& body)
{
    return build_loop_with_teardown(loops, unroll,
        [&](a64::Assembler& a) { mrs_fpcr(a, x20); setup(a); },
        body,
        [](a64::Assembler& a)  { msr_fpcr(a, x20); });
}

// x21 = x20 ^ bits: the guest value of a test that flips `bits`.
static void set_guest(a64::Assembler& a, uint64_t bits) {
    a.mov(x21, Imm(bits));
    a.eor(x21, x20, x21);
}

// FPCR as the process runs with it, for the section note.
static uint64_t read_fpcr() {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);
    mrs_fpcr(a, x0);
    a.ret(x30);
    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) return 0;
    const uint64_t v = reinterpret_cast<uint64_t (*)()>(fn)();
    g_jit_pool->release(fn);
    return v;
}

// ════════════════════════════════════════════════════════════════════════════
// Section 1: MRS FPCR / MSR FPCR on their own
// ════════════════════════════════════════════════════════════════════════════
//
// MRS has no register input, so it cannot be chained through its own result;
// "MRS FPCR + ADD chain" consumes every result in a 1-clk ADD chain, which
// reads 1 clk per pair unless MRS issues slower than one per cycle. The only
// true dependency through FPCR is MRS → MSR → MRS, the round-trip row: on a
// core that renames FPCR it is MRS latency plus MSR latency, on a core that
// serialises the write it is the cost of the drain.
//
// The toggle rows alternate between the host value and the host value with
// the named bits flipped, so every write changes FPCR. A core may special-
// case a write that leaves FPCR unchanged (compare "same value").
//
// "check" is the lazy alternative to an unconditional write: read FPCR,
// compare with the wanted value, branch over the MSR. The values always
// match, so the MSR never executes and the branch predicts perfectly.
//
// M5: MRS 1 per clk; MSR of the value already there 11 clk (zero or not);
// MSR of any different value 34 clk, whichever bit changes, which is what
// an ISB costs on this core. The check idiom is 1 clk.

static constexpr uint64_t kSysregLoops  = 1'000'000;
static constexpr uint32_t kSysregUnroll = 16;

static void run_fpcr_toggle(const BenchmarkParams& base, const char* what, uint64_t bits) {
    const uint64_t loops = scale_loops(kSysregLoops);
    auto fn = build_fpenv_loop(loops, kSysregUnroll,
        [=](a64::Assembler& a) { set_guest(a, bits); },
        [](a64::Assembler& a, uint32_t u) { msr_fpcr(a, (u & 1) ? x20 : x21); });
    char name[80];
    snprintf(name, sizeof(name), "MSR FPCR tput, toggle %s", what);
    run_one(name, fn, params_for(base, loops, kSysregUnroll));
}

static void run_fpcr_access_tests(const BenchmarkParams& base) {
    section("FPCR read / write (MRS FPCR, MSR FPCR)");
    const bool afp = cpu_has(CpuFeature::AFP);
    printf("  clk per MRS or MSR unless the row says otherwise. FPCR on entry: 0x%08llx.\n"
           "  Toggle rows alternate the entry value with the named bits flipped, so\n"
           "  every write changes FPCR. RMode toggles RN/RZ. FIZ, AH, NEP are FEAT_AFP.\n",
           static_cast<unsigned long long>(read_fpcr()));
    if (!afp) skip_feature(CpuFeature::AFP, "FIZ / AH / NEP rows");
    printf("\n");

    const uint64_t loops  = scale_loops(kSysregLoops);
    const uint32_t unroll = kSysregUnroll;

    // ── MRS FPCR ──────────────────────────────────────────────────────────
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t u) { mrs_fpcr(a, xr(u % 8)); });
        run_one("MRS FPCR tput", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll,
            [](a64::Assembler& a) { a.mov(x0, Imm(0)); },
            [](a64::Assembler& a, uint32_t) {
                mrs_fpcr(a, x1);
                a.add(x0, x0, x1);
            });
        run_one("MRS FPCR + ADD chain (per pair)", fn, params_for(base, loops, unroll));
    }

    // ── MSR FPCR ──────────────────────────────────────────────────────────
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { msr_fpcr(a, x20); });
        run_one("MSR FPCR tput, same value", fn, params_for(base, loops, unroll));
    }
    {
        // The same again with a non-default value in place, in case "unchanged"
        // is only cheap when FPCR is zero.
        auto fn = build_fpenv_loop(loops, unroll,
            [](a64::Assembler& a) { set_guest(a, kFpcrFZ); msr_fpcr(a, x21); },
            [](a64::Assembler& a, uint32_t) { msr_fpcr(a, x21); });
        run_one("MSR FPCR tput, same value (FZ set)", fn, params_for(base, loops, unroll));
    }
    run_fpcr_toggle(base, "FZ",    kFpcrFZ);
    run_fpcr_toggle(base, "RMode", kFpcrRZ);
    if (afp) {
        run_fpcr_toggle(base, "FIZ",        kFpcrFIZ);
        run_fpcr_toggle(base, "AH",         kFpcrAH);
        run_fpcr_toggle(base, "NEP",        kFpcrNEP);
        run_fpcr_toggle(base, "AH|NEP|FIZ", kFpcrAH | kFpcrNEP | kFpcrFIZ);
    }

    // ── Through FPCR ──────────────────────────────────────────────────────
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) {
                mrs_fpcr(a, x0);
                msr_fpcr(a, x0);
            });
        run_one("MRS->MSR FPCR chain (per round trip)", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) {
                Label skip = a.new_label();
                mrs_fpcr(a, x1);
                a.cmp(x1, x20);
                a.b_eq(skip);
                msr_fpcr(a, x20);
                a.bind(skip);
            });
        run_one("MRS FPCR; CMP; B.EQ over MSR (per check)", fn, params_for(base, loops, unroll));
    }
    {
        // What a full pipeline drain costs on this core, for scale.
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { a.isb(Imm(15)); });
        run_one("ISB tput (reference)", fn, params_for(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// FP operations the remaining sections interleave with system-register access
// ════════════════════════════════════════════════════════════════════════════
//
// Chain registers are vr(0..15) (v0–v7, v16–v23), the constant lives in
// vr(16) (v24). The constants make every operation inexact without ever
// leaving the normal range, so rounding-mode changes do real work and FPSR
// accumulates IXC the way ordinary code does:
//   FADD  acc += 0.1            (f32 lanes stop growing at 2^21; still inexact)
//   FMUL  acc *= 1 + ulp-ish    (e^20 at most over one call)

//   exact FADD  acc += 0.0      (raises nothing: FPSR stays as it was)

struct FpOp {
    const char* label;
    bool        vec;       // 4×f32 vector, else f64 scalar
    bool        mul;
    bool        exact = false;
};
static constexpr FpOp kFaddD{ "FADD f64",   false, false };
static constexpr FpOp kFmulD{ "FMUL f64",   false, true  };
static constexpr FpOp kFaddV{ "FADD v4f32", true,  false };
static constexpr FpOp kFmulV{ "FMUL v4f32", true,  true  };
static constexpr FpOp kFaddDExact{ "FADD f64 exact", false, false, true };

static constexpr uint32_t kFpConst = 16;   // vr(16) = v24

static void fp_load(a64::Assembler& a, const FpOp& op, uint32_t reg, double f64v, float f32v) {
    if (op.vec) {
        uint32_t bits;
        memcpy(&bits, &f32v, sizeof(bits));
        a.mov(w9, Imm(bits));
        a.dup(vr(reg).s4(), w9);
    } else {
        uint64_t bits;
        memcpy(&bits, &f64v, sizeof(bits));
        a.mov(x9, Imm(bits));
        a.fmov(vr(reg).d(), x9);
    }
}

// Seeds chain registers 0..nregs-1 and the constant. Clobbers x9.
static void fp_seed(a64::Assembler& a, const FpOp& op, uint32_t nregs) {
    if (op.mul)        fp_load(a, op, kFpConst, 1.0000001, 1.0000001f);
    else if (op.exact) fp_load(a, op, kFpConst, 0.0, 0.0f);
    else               fp_load(a, op, kFpConst, 0.1, 0.1f);
    for (uint32_t i = 0; i < nregs; ++i) fp_load(a, op, i, 1.5, 1.5f);
}

// reg = reg OP constant.
static void fp_emit(a64::Assembler& a, const FpOp& op, uint32_t reg) {
    if (op.vec) {
        const Vec d = vr(reg).s4(), c = vr(kFpConst).s4();
        if (op.mul) a.fmul(d, d, c); else a.fadd(d, d, c);
    } else {
        const Vec d = vr(reg).d(), c = vr(kFpConst).d();
        if (op.mul) a.fmul(d, d, c); else a.fadd(d, d, c);
    }
}

// ── FP code with something every N operations ─────────────────────────────────
//
// One loop iteration is `groups` groups of (N FP ops, then between(a, g)).
// The FP ops rotate over `chains` registers: 1 makes one dependency chain
// (latency-bound, the FP unit mostly idle), 16 makes the loop throughput-
// bound (M5's 2-clk FADD still reads 0.38 clk with 8 chains, 0.25 with 16). Results are per FP operation, so the row without the system-
// register access is the baseline and (row − baseline) × N is what one
// access costs the surrounding code. groups is even so a between() that
// alternates two values ends each iteration where it started.

// FP ops per timed call: fewer when nearly every other instruction is a
// system-register access that may cost tens of cycles.
static constexpr uint64_t kInterleaveOps      = 64'000'000;
static constexpr uint64_t kInterleaveOpsDense = 16'000'000;   // N <= 4

template<class FBetween>
static void run_interleaved(const BenchmarkParams& base, const FpOp& op, uint32_t chains,
                            uint32_t every, const char* what, uint64_t guest_bits,
                            FBetween&& between)
{
    const uint32_t n      = every ? every : 64;
    const uint32_t groups = n >= 32 ? 2 : 64 / n;
    const uint32_t ops    = groups * n;
    uint64_t loops = scale_loops(every && every <= 4 ? kInterleaveOpsDense : kInterleaveOps) / ops;
    if (loops == 0) loops = 1;

    auto fn = build_fpenv_loop(loops, groups,
        [&](a64::Assembler& a) {
            set_guest(a, guest_bits);
            fp_seed(a, op, chains);
        },
        [&](a64::Assembler& a, uint32_t g) {
            for (uint32_t i = 0; i < n; ++i) fp_emit(a, op, (g * n + i) % chains);
            if (every) between(a, g);
        });

    char shape[24], name[96];
    if (chains == 1) snprintf(shape, sizeof(shape), "chain");
    else             snprintf(shape, sizeof(shape), "%u chains", chains);
    if (every) snprintf(name, sizeof(name), "%s %s, %s every %u", op.label, shape, what, every);
    else       snprintf(name, sizeof(name), "%s %s, %s", op.label, shape, what);
    run_one(name, fn, params_for(base, loops, ops));
}

static constexpr uint32_t kEvery[] = { 1, 4, 16, 64 };

// ════════════════════════════════════════════════════════════════════════════
// Section 2: MSR FPCR inside FP code
// ════════════════════════════════════════════════════════════════════════════
//
// Does a write stall the FP instructions around it? Two shapes:
//
//   chain      one FADD/FMUL dependency chain with an MSR FPCR every N ops.
//              If FPCR were renamed and the FP ops simply took the new value
//              as an input, the chain would not notice.
//   16 chains  sixteen independent chains, MSR every N ops. This is the
//              shape that shows a drain: the out-of-order overlap is lost at
//              every write.
//
// "same" rewrites the value already there; "toggle" flips FZ on every
// write, host value and host^FZ alternately.
//
// Then the translator's pattern itself, per trip:
//     MSR FPCR, guest ; k × FADD v4f32 ; MSR FPCR, host
// with the k ops on one chain ("chained") or on k separate registers
// ("indep"), against the same k ops with no writes. "same" writes the host
// value both times (a guest whose mode equals the host's).
//
// M5: a write that changes FPCR costs the surrounding code 32–36 clk in
// either shape (+0.5 clk per op at N = 64), so a bracket is 67–71 clk per
// trip on top of its FP ops, for FZ, RZ and AH|NEP alike. A write of the
// unchanged value limits the loop to one write per 11 clk but does not stop
// FP ops issuing around it: 16 chains with one every 64 FADDs, or one chain
// with one every 16, read the same as with none, and a "same" bracket
// around 16 chained FADDs costs 0.6 clk.

static void run_fpcr_interleave(const BenchmarkParams& base, const FpOp& op, uint32_t chains) {
    run_interleaved(base, op, chains, 0, "no MSR", 0, [](a64::Assembler&, uint32_t) {});
    for (const uint32_t n : kEvery)
        run_interleaved(base, op, chains, n, "MSR FPCR same", 0,
            [](a64::Assembler& a, uint32_t) { msr_fpcr(a, x20); });
    for (const uint32_t n : kEvery)
        run_interleaved(base, op, chains, n, "MSR FPCR toggle", kFpcrFZ,
            [](a64::Assembler& a, uint32_t g) { msr_fpcr(a, (g & 1) ? x20 : x21); });
}

static constexpr uint64_t kBracketTrips = 4'000'000;   // trips per timed call

// guest == nullptr: no writes at all.
static void run_fpcr_bracket(const BenchmarkParams& base, const char* guest, uint64_t guest_bits,
                             uint32_t k, bool chained)
{
    const uint32_t trips = k >= 16 ? 2 : 32 / k / 2;          // per iteration
    uint64_t loops = scale_loops(kBracketTrips) / trips;
    if (loops == 0) loops = 1;
    const uint32_t nregs = chained ? 1 : k;

    auto fn = build_fpenv_loop(loops, trips,
        [&](a64::Assembler& a) {
            set_guest(a, guest_bits);
            fp_seed(a, kFaddV, nregs);
        },
        [&](a64::Assembler& a, uint32_t) {
            if (guest) msr_fpcr(a, x21);
            for (uint32_t i = 0; i < k; ++i) fp_emit(a, kFaddV, i % nregs);
            if (guest) msr_fpcr(a, x20);
        });

    char name[96];
    if (guest)
        snprintf(name, sizeof(name), "FPCR bracket (%s), %u FADD v4f32 %s",
                 guest, k, chained ? "chained" : "indep");
    else
        snprintf(name, sizeof(name), "no bracket, %u FADD v4f32 %s",
                 k, chained ? "chained" : "indep");
    run_one(name, fn, params_for(base, loops, trips));
}

static void run_fpcr_in_fp_code_tests(const BenchmarkParams& base) {
    section("FPCR writes inside FP code");
    const bool afp = cpu_has(CpuFeature::AFP);
    printf("  clk per FP operation, with an MSR FPCR after every N of them. (row - the\n"
           "  \"no MSR\" row) x N is what one write costs the code around it. \"chain\" is\n"
           "  one dependency chain, \"16 chains\" is throughput-bound. \"same\" rewrites the\n"
           "  current value, \"toggle\" flips FZ on every write.\n\n");

    for (const FpOp* op : { &kFaddD, &kFmulD, &kFaddV, &kFmulV })
        run_fpcr_interleave(base, *op, 1);
    for (const FpOp* op : { &kFaddD, &kFaddV })
        run_fpcr_interleave(base, *op, 16);

    printf("\n  Per trip: MSR FPCR, guest; k x FADD v4f32; MSR FPCR, host. \"chained\" puts\n"
           "  the k ops on one dependency chain, \"indep\" on k registers. \"same\" writes\n"
           "  the host value both times.\n");
    if (!afp) skip_feature(CpuFeature::AFP, "the AH|NEP bracket");
    printf("\n");

    for (const bool chained : { true, false }) {
        for (const uint32_t k : { 1u, 4u, 16u }) {
            run_fpcr_bracket(base, nullptr, 0, k, chained);
            run_fpcr_bracket(base, "same",  0, k, chained);
            run_fpcr_bracket(base, "FZ",    kFpcrFZ, k, chained);
            run_fpcr_bracket(base, "RZ",    kFpcrRZ, k, chained);
            if (afp)
                run_fpcr_bracket(base, "AH|NEP", kFpcrAH | kFpcrNEP, k, chained);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: MRS FPSR / MSR FPSR
// ════════════════════════════════════════════════════════════════════════════
//
// FPSR holds the cumulative exception flags, which every FP instruction may
// set. A core either renames the flags like NZCV or merges them at
// retirement; in the second case MRS FPSR has to wait for every FP op ahead
// of it to retire, and MSR FPSR (clearing the flags) has to be ordered
// against them too. The interleaved rows show what that does to the FP code:
// the ops are inexact, so each one raises IXC, and after a clear the next op
// sets a flag that was not set before. The "exact" rows repeat the clear
// with operations that raise nothing (acc + 0.0), to tell "the write is
// slow" from "raising a flag that was clear is slow".
//
// The round-trip rows put FPSR on a real dependency chain. One trip is
//     AND x3, x0, x5 (= 0) ; ADD x3, x3, x4 (= bits of 1.5) ; FMOV d1, x3 ;
//     FADD d0, d1, d24 ; <x0 from the FADD>
// where the last step is FMOV x0, d0 in the reference row and MRS x0, FPSR in
// the FPSR rows (the AND discards the value; only the dependency matters).
// FPSR row − reference row is what reading the flags of an FP op costs over
// reading its result. The third row clears FPSR before each FADD: the
// per-instruction "clear; op; read flags" pattern.
//
// M5: MRS FPSR is one per 10 clk (MRS FPCR: one per clk) but, like an
// unchanged FPCR write, does not hold up FP ops around it: no cost at one
// read per 16 chained or 64 independent FADDs. Clearing is 12 clk, and the
// write is not the expensive part: with exact ops a clear every 64 is free,
// with inexact ops it costs 41 clk (chain) to 65–75 clk (16 chains). The
// first operation that raises a flag which is currently clear pays a
// pipeline flush. On the chain, flags cost 10.7 clk more than the result
// (25.2 vs 14.5 per trip), and clear; op; read is 57.8 clk per trip.

static void run_fpsr_interleave(const BenchmarkParams& base, const FpOp& op, uint32_t chains,
                                bool mrs, bool msr)
{
    run_interleaved(base, op, chains, 0, "no FPSR access", 0, [](a64::Assembler&, uint32_t) {});
    if (mrs)
        for (const uint32_t n : kEvery)
            run_interleaved(base, op, chains, n, "MRS FPSR", 0,
                [](a64::Assembler& a, uint32_t) { mrs_fpsr(a, x1); });
    if (msr)
        for (const uint32_t n : kEvery)
            run_interleaved(base, op, chains, n, "MSR FPSR", 0,
                [](a64::Assembler& a, uint32_t) { msr_fpsr(a, xzr); });
    if (mrs && msr)
        for (const uint32_t n : kEvery)
            run_interleaved(base, op, chains, n, "MRS+MSR FPSR", 0,
                [](a64::Assembler& a, uint32_t) { mrs_fpsr(a, x1); msr_fpsr(a, xzr); });
}

enum class FpsrTrip { FmovRef, Mrs, ClearThenMrs };

static void run_fpsr_round_trip(const BenchmarkParams& base, const char* name, FpsrTrip t) {
    const uint32_t unroll = 8;
    const uint64_t loops  = scale_loops(kSysregLoops);
    auto fn = build_fpenv_loop(loops, unroll,
        [](a64::Assembler& a) {
            fp_seed(a, kFaddD, 1);
            uint64_t bits;
            const double v = 1.5;
            memcpy(&bits, &v, sizeof(bits));
            a.mov(x4, Imm(bits));
            a.mov(x5, Imm(0));
            a.mov(x0, Imm(0));
        },
        [=](a64::Assembler& a, uint32_t) {
            if (t == FpsrTrip::ClearThenMrs) msr_fpsr(a, xzr);
            a.and_(x3, x0, x5);
            a.add(x3, x3, x4);
            a.fmov(d1, x3);
            a.fadd(d0, d1, vr(kFpConst).d());
            if (t == FpsrTrip::FmovRef) a.fmov(x0, d0);
            else                        mrs_fpsr(a, x0);
        });
    run_one(name, fn, params_for(base, loops, unroll));
}

static void run_fpsr_tests(const BenchmarkParams& base) {
    section("FPSR read / write (MRS FPSR, MSR FPSR)");
    printf("  clk per MRS or MSR unless the row says otherwise. MSR FPSR writes xzr\n"
           "  (clears the cumulative flags) except in the toggle row (0 / IXC).\n\n");

    const uint64_t loops  = scale_loops(kSysregLoops);
    const uint32_t unroll = kSysregUnroll;

    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t u) { mrs_fpsr(a, xr(u % 8)); });
        run_one("MRS FPSR tput", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll,
            [](a64::Assembler& a) { a.mov(x0, Imm(0)); },
            [](a64::Assembler& a, uint32_t) {
                mrs_fpsr(a, x1);
                a.add(x0, x0, x1);
            });
        run_one("MRS FPSR + ADD chain (per pair)", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) { msr_fpsr(a, xzr); });
        run_one("MSR FPSR tput, xzr (clear)", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll,
            [](a64::Assembler& a) { a.mov(x1, Imm(kFpsrIXC)); },
            [](a64::Assembler& a, uint32_t u) { msr_fpsr(a, (u & 1) ? xzr : x1); });
        run_one("MSR FPSR tput, toggle IXC", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t) {
                mrs_fpsr(a, x0);
                msr_fpsr(a, x0);
            });
        run_one("MRS->MSR FPSR chain (per round trip)", fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_fpenv_loop(loops, unroll, no_setup,
            [](a64::Assembler& a, uint32_t u) {
                msr_fpsr(a, xzr);
                mrs_fpsr(a, xr(u % 8));
            });
        run_one("MSR FPSR,xzr; MRS FPSR (per pair)", fn, params_for(base, loops, unroll));
    }

    printf("\n  clk per FP operation, with an FPSR access after every N of them, as in the\n"
           "  FPCR section. The FP ops are inexact (each raises IXC) except in the\n"
           "  \"exact\" rows, which raise nothing.\n\n");
    for (const uint32_t chains : { 1u, 16u }) {
        run_fpsr_interleave(base, kFaddD, chains, true, true);
        run_fpsr_interleave(base, kFaddV, chains, true, true);
        run_fpsr_interleave(base, kFaddDExact, chains, false, true);
    }

    printf("\n  FPSR on a dependency chain, clk per trip: x -> FMOV -> FADD f64 -> x, the\n"
           "  last step FMOV x,d (ref) or MRS FPSR. FPSR row - ref = reading an FP op's\n"
           "  flags instead of its result.\n\n");
    run_fpsr_round_trip(base, "x->FMOV->FADD->FMOV->x (per trip, ref)",  FpsrTrip::FmovRef);
    run_fpsr_round_trip(base, "x->FMOV->FADD->MRS FPSR->x (per trip)",   FpsrTrip::Mrs);
    run_fpsr_round_trip(base, "MSR FPSR,xzr; x->FMOV->FADD->MRS FPSR->x", FpsrTrip::ClearThenMrs);
}

// ── Public entry point ────────────────────────────────────────────────────────

void run_fpenv_tests(const BenchmarkParams& base_params) {
    run_fpcr_access_tests(base_params);
    run_fpcr_in_fp_code_tests(base_params);
    run_fpsr_tests(base_params);
}

} // namespace arm64bench::gen
