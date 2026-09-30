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

// ── Public entry point ────────────────────────────────────────────────────────

void run_fpenv_tests(const BenchmarkParams& base_params) {
    run_fpcr_access_tests(base_params);
}

} // namespace arm64bench::gen
