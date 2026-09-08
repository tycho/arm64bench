// selftest.cpp
// Verifies the measurement machinery that every arm64bench result depends on.
//
// This is deliberately NOT a microarchitecture test. It asks only questions
// whose answers are the same on every ARM64 core and every supported OS:
//
//   timer         — monotonic, sane frequency, sub-microsecond step, and a
//                   10 ms sleep measures as roughly 10 ms of wall time
//   cycle counter — (when accessible) monotonic, plausible implied clock,
//                   and a chained-ADD loop costs ~1 cycle per instruction
//   calibration   — produces a plausible CPU frequency
//   harness       — calls the test function exactly as documented, honours
//                   smoke mode and the name filter, and reports a chained
//                   ADD at ~1 clk with a sane min/median relationship
//   JIT pool      — compile/release churn does not fail
//   CPU features  — every feature the generators can gate on is reported,
//                   so a CI log shows what the runner actually has
//
// Checks that need a resource the host does not grant (a PMU on a VM, say)
// are reported as SKIP, not FAIL. The process exit code is the number of
// failed checks, so `ctest` and CI treat any failure as red.
//
// Two kinds of check: correctness (call counts, monotonic clocks, feature
// implications, result-structure invariants) and measurement quality (a
// chained ADD reads ~1 clk, sleeps are not absurdly long, calibration is
// plausible). On a shared, oversubscribed CI VM the quality checks can fail
// with nothing wrong in the code — a runner where 10 ms sleeps take 50 ms
// makes the reference probes slower than the test and drags ratios below
// 1. With --lenient (or ARM64BENCH_SELFTEST_LENIENT=1) quality checks print
// WARN instead of FAIL and do not affect the exit code.
//
// Usage: arm64bench_selftest [--verbose] [--lenient]

#include "harness.h"
#include "timer.h"
#include "cycle_counter.h"
#include "cpu_features.h"
#include "jit_buffer.h"
#include "gen_integer.h"

#include <asmjit/core.h>
#include <asmjit/a64.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

using namespace arm64bench;
using namespace asmjit;
using namespace asmjit::a64;

// ── Minimal check framework ──────────────────────────────────────────────────

static int  s_failed  = 0;
static int  s_passed  = 0;
static int  s_skipped = 0;
static int  s_warned  = 0;
static bool s_verbose = false;
static bool s_lenient = false;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            ++s_passed;                                                      \
            if (s_verbose) { printf("    pass: "); printf(__VA_ARGS__); printf("\n"); } \
        } else {                                                             \
            ++s_failed;                                                      \
            printf("    FAIL: "); printf(__VA_ARGS__); printf("\n");         \
        }                                                                    \
        fflush(stdout);                                                      \
    } while (0)

// Measurement-quality check: FAIL normally, WARN under --lenient.
#define CHECK_Q(cond, ...)                                                   \
    do {                                                                     \
        if (cond) {                                                          \
            ++s_passed;                                                      \
            if (s_verbose) { printf("    pass: "); printf(__VA_ARGS__); printf("\n"); } \
        } else if (s_lenient) {                                              \
            ++s_warned;                                                      \
            printf("    WARN: "); printf(__VA_ARGS__); printf("  [lenient]\n"); \
        } else {                                                             \
            ++s_failed;                                                      \
            printf("    FAIL: "); printf(__VA_ARGS__); printf("\n");         \
        }                                                                    \
        fflush(stdout);                                                      \
    } while (0)

static void skip(const char* why) {
    ++s_skipped;
    printf("    SKIP: %s\n", why);
    fflush(stdout);
}

static void section(const char* name) {
    printf("[%s]\n", name);
    fflush(stdout);
}

static void note(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("    ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    fflush(stdout);
}

static void sleep_ms(uint32_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Busy-wait for `ns` nanoseconds of wall time. Returns the wall time actually
// spent (ns), which is ≥ ns by at most one timer step.
static double spin_ns(uint64_t ns) {
    const RawTick t0 = tick_now();
    volatile uint64_t v = 1;
    while (ticks_to_ns(tick_now() - t0) < ns)
        v ^= v * 6364136223846793005ULL + 1442695040888963407ULL;
    (void)v;
    return ticks_to_ns_f(tick_now() - t0);
}

// ── JIT helpers ──────────────────────────────────────────────────────────────

// Increments the uint64 at `counter` by one. Used to count harness calls.
static JitPool::TestFn build_counter_fn(uint64_t* counter) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);
    a.mov(x0, Imm(reinterpret_cast<uint64_t>(counter)));
    a.ldr(x1, ptr(x0));
    a.add(x1, x1, Imm(1));
    a.str(x1, ptr(x0));
    a.ret(x30);
    return g_jit_pool->compile(code);
}

// `chains` independent ADD dependency chains (x0..x[chains-1] += x20),
// `unroll` instructions per iteration, `loops` iterations.
// chains == 1 is the latency chain; chains ≥ 2 exposes throughput.
static JitPool::TestFn build_add_loop(uint64_t loops, uint32_t unroll, uint32_t chains) {
    static const Gp kChainRegs[] = { x0, x1, x2, x3, x4, x5, x6, x7 };
    if (chains == 0 || chains > 8) return nullptr;

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));
    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(1));
    for (uint32_t c = 0; c < chains; ++c)
        a.mov(kChainRegs[c], Imm(c + 1));

    a.align(AlignMode::kCode, 64);
    Label top = a.new_label();
    a.bind(top);
    for (uint32_t u = 0; u < unroll; ++u) {
        const Gp& r = kChainRegs[u % chains];
        a.add(r, r, x20);
    }
    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, top);

    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);
    return g_jit_pool->compile(code);
}

// ── Tests ────────────────────────────────────────────────────────────────────

static void test_timer_basics() {
    section("timer: frequency and conversions");

    const uint64_t freq = tick_frequency();
    CHECK(freq > 0, "tick_frequency() = %llu Hz", (unsigned long long)freq);
#if !defined(_WIN32)
    CHECK(freq == 1'000'000'000ULL, "POSIX tick frequency is 1 GHz (ns)");
#endif
    // One second of ticks must convert to exactly 1e9 ns (integer path) and
    // to 1e9 within rounding (double path).
    CHECK(ticks_to_ns(freq) == 1'000'000'000ULL,
          "ticks_to_ns(1 s) = %llu", (unsigned long long)ticks_to_ns(freq));
    CHECK(std::fabs(ticks_to_ns_f(freq) - 1e9) < 1.0,
          "ticks_to_ns_f(1 s) = %.1f", ticks_to_ns_f(freq));
    // Large-delta overflow guard: 1000 s of ticks.
    CHECK(ticks_to_ns(freq * 1000) == 1'000'000'000'000ULL,
          "ticks_to_ns(1000 s) = %llu", (unsigned long long)ticks_to_ns(freq * 1000));
}

static void test_timer_monotonic() {
    section("timer: monotonicity and step");

    static constexpr uint32_t kReads = 2'000'000;
    RawTick prev = tick_now();
    uint32_t backwards = 0, advanced = 0;
    for (uint32_t i = 0; i < kReads; ++i) {
        const RawTick now = tick_now();
        if (now < prev)      ++backwards;
        else if (now > prev) ++advanced;
        prev = now;
    }
    CHECK(backwards == 0, "tick_now() never went backwards over %u reads (%u violations)",
          kReads, backwards);
    CHECK(advanced > 0, "tick_now() advanced %u times over %u reads", advanced, kReads);

    // wait_for_tick() must return a value strictly greater than the value
    // read just before it, and the step it observes should be tiny.
    uint64_t min_step = UINT64_MAX;
    bool strictly_after = true;
    for (uint32_t i = 0; i < 1000; ++i) {
        const RawTick before  = tick_now();
        const RawTick aligned = wait_for_tick();
        if (aligned <= before) strictly_after = false;
        const RawTick next    = wait_for_tick();
        const uint64_t step   = ticks_to_ns(next - aligned);
        if (step < min_step) min_step = step;
    }
    CHECK(strictly_after, "wait_for_tick() always returns after the preceding read");
    CHECK(min_step > 0 && min_step <= 1'000'000,
          "minimum observed timer step = %llu ns (expect >0, <=1 ms)",
          (unsigned long long)min_step);
}

static void test_sleep_wall_time() {
    section("timer: 10 ms sleeps measure as ~10 ms");

    static constexpr int      kSleeps   = 5;
    static constexpr uint32_t kSleepMs  = 10;
    double measured[kSleeps];
    double total = 0.0;
    for (int i = 0; i < kSleeps; ++i) {
        const RawTick t0 = tick_now();
        sleep_ms(kSleepMs);
        measured[i] = ticks_to_ns_f(tick_now() - t0) * 1e-6;
        total += measured[i];
    }
    note("sleep(10 ms) measured: %.2f %.2f %.2f %.2f %.2f ms",
         measured[0], measured[1], measured[2], measured[3], measured[4]);

    // Lower bound is strict: a sleep that returns early means the clock is
    // running fast or the sleep is broken. Windows' default 15.6 ms timer
    // tick can legitimately stretch a 10 ms sleep to ~26 ms, and a loaded
    // CI VM can add more, so the upper bound is deliberately loose.
    bool none_short = true, none_absurd = true;
    for (int i = 0; i < kSleeps; ++i) {
        if (measured[i] < 9.0)   none_short  = false;
        if (measured[i] > 200.0) none_absurd = false;
    }
    CHECK(none_short,  "no sleep returned early (all >= 9 ms)");
    CHECK_Q(none_absurd, "no sleep took more than 200 ms");
    CHECK_Q(total < 500.0, "5 x 10 ms sleeps took %.1f ms total (< 500 ms)", total);
}

// PMU-implied clock from a 10 ms spin, or 0 if the PMU is unavailable.
static double s_pmu_spin_ghz = 0.0;

static void test_cycle_counter() {
    section("cycle counter (PMU)");

    const bool ok = cycle_counter_init();
    CHECK(cycle_counter_available() == ok,
          "cycle_counter_available() agrees with cycle_counter_init() (%s)",
          ok ? "available" : "unavailable");
    CHECK(cycle_counter_init() == ok, "cycle_counter_init() is idempotent");

    if (!ok) {
        CHECK(cycle_counter_read() == 0, "cycle_counter_read() returns 0 when unavailable");
#if defined(__APPLE__)
        skip("PMU not accessible (macOS 15+ needs root: sudo ./arm64bench_selftest; VMs have no PMU)");
#elif defined(_WIN32)
        skip("PMCCNTR_EL0 not readable from EL0 (no PMU driver enabling user access, or a VM)");
#else
        skip("perf_event_open refused (no PMU in this VM, or perf_event_paranoid too strict)");
#endif
        return;
    }

    // Monotonic across consecutive reads.
    uint64_t prev = cycle_counter_read();
    uint32_t backwards = 0;
    for (uint32_t i = 0; i < 10'000; ++i) {
        const uint64_t now = cycle_counter_read();
        if (now < prev) ++backwards;
        prev = now;
    }
    CHECK(backwards == 0, "cycle_counter_read() never went backwards (%u violations)", backwards);

    // Implied clock over a 10 ms busy spin. Any real ARM64 core at any
    // P-state lands comfortably inside 0.3–8 GHz.
    {
        const uint64_t c0 = cycle_counter_read();
        const double   wall_ns = spin_ns(10'000'000);
        const uint64_t c1 = cycle_counter_read();
        s_pmu_spin_ghz = static_cast<double>(c1 - c0) / wall_ns;
        CHECK_Q(s_pmu_spin_ghz > 0.3 && s_pmu_spin_ghz < 8.0,
              "10 ms spin implies %.3f GHz (expect 0.3-8 GHz)", s_pmu_spin_ghz);
    }

    // Cycles charged while sleeping. Per-thread counters (macOS kpc) must
    // charge far less for a sleep than for a spin of equal wall time; a
    // per-CPU counter (Windows PMCCNTR_EL0) may count the idle core, so that
    // is reported but not asserted.
    {
        const uint64_t c0 = cycle_counter_read();
        sleep_ms(10);
        const uint64_t c1 = cycle_counter_read();
        const double sleep_ghz = static_cast<double>(c1 - c0) / 10e6;
        note("10 ms sleep charged %.3f GHz-equivalent of cycles", sleep_ghz);
#if defined(__APPLE__)
        CHECK_Q(sleep_ghz < s_pmu_spin_ghz * 0.5,
              "per-thread counter charges < 50%% of spin rate while sleeping");
#endif
    }

    // A chained ADD costs exactly 1 cycle on every ARMv8 implementation.
    // 16M instructions swamp the read overhead (a syscall on macOS).
    {
        static constexpr uint64_t kLoops  = 500'000;
        static constexpr uint32_t kUnroll = 32;
        JitPool::TestFn fn = build_add_loop(kLoops, kUnroll, 1);
        CHECK(fn != nullptr, "compiled ADD latency chain");
        if (fn) {
            fn();   // warm
            double best = 1e9;
            for (int i = 0; i < 5; ++i) {
                const uint64_t c0 = cycle_counter_read();
                fn();
                const uint64_t c1 = cycle_counter_read();
                const double cpi = static_cast<double>(c1 - c0) / (kLoops * kUnroll);
                if (cpi < best) best = cpi;
            }
            CHECK_Q(best > 0.90 && best < 1.25,
                  "chained ADD = %.3f cycles/insn by PMU (expect ~1.0)", best);
            g_jit_pool->release(fn);
        }
    }
}

static void test_calibration() {
    section("CPU frequency calibration");

    const uint64_t hz = calibrate_cpu_freq();
    CHECK(hz == g_cpu_freq_hz, "calibrate_cpu_freq() stores its result in g_cpu_freq_hz");
    CHECK_Q(hz >= 300'000'000ULL && hz <= 8'000'000'000ULL,
          "calibrated %.0f MHz (expect 300-8000 MHz)", hz / 1e6);

    // Sanity: ticks_to_cycles() uses the calibrated value.
    const double cyc = ticks_to_cycles(tick_frequency());  // one second
    CHECK(std::fabs(cyc - static_cast<double>(hz)) < 1.0,
          "ticks_to_cycles(1 s) = %.0f cycles", cyc);

    if (s_pmu_spin_ghz > 0.0) {
        // Informational only: the spin and the calibration may have run on
        // cores with different clocks (P vs E) or at different P-states.
        note("PMU-implied clock %.3f GHz vs calibrated %.3f GHz (ratio %.3f)",
             s_pmu_spin_ghz, hz / 1e9, s_pmu_spin_ghz / (hz / 1e9));
    }
}

static void test_harness_call_accounting() {
    section("harness: call accounting, smoke mode, name filter");

    static uint64_t counter = 0;
    JitPool::TestFn fn = build_counter_fn(&counter);
    CHECK(fn != nullptr, "compiled counter function");
    if (!fn) return;

    counter = 0;
    fn();
    CHECK(counter == 1, "counter function increments by exactly 1 per call (%llu)",
          (unsigned long long)counter);

    BenchmarkParams p{};
    p.loops                 = 1;
    p.instructions_per_loop = 1;
    p.num_samples           = 5;
    p.num_warmup            = 2;
    p.num_per_sample_warmup = 1;
    p.inter_sample_ms       = 0;
    p.discard_highest       = 1;

    // No reference registered: exactly warmup + samples × (per-sample warmup + 1).
    set_reference_function(ReferenceParams{});
    counter = 0;
    benchmark(fn, "selftest counter (no ref)", p);
    const uint64_t expected = p.num_warmup + p.num_samples * (p.num_per_sample_warmup + 1);
    CHECK(counter == expected, "measure mode without reference: %llu calls (expect %llu)",
          (unsigned long long)counter, (unsigned long long)expected);

    // With a reference registered, retries can add calls but never remove them.
    {
        ReferenceParams ref;
        ref.fn          = gen::create_add_latency_ref(20'000, 32);
        ref.total_insns = 20'000ULL * 32;
        CHECK(ref.fn != nullptr, "compiled ADD reference function");
        set_reference_function(ref);
        counter = 0;
        benchmark(fn, "selftest counter (with ref)", p);
        CHECK(counter >= expected && counter <= expected + p.num_samples * ref.retry_limit,
              "measure mode with reference: %llu calls (expect %llu..%llu)",
              (unsigned long long)counter, (unsigned long long)expected,
              (unsigned long long)(expected + p.num_samples * ref.retry_limit));
        set_reference_function(ReferenceParams{});
        g_jit_pool->release(ref.fn);
    }

    // Smoke mode: exactly one call, zeroed result, counted.
    {
        const uint32_t before = smoke_test_count();
        set_run_mode(RunMode::Smoke);
        CHECK(run_mode() == RunMode::Smoke, "set_run_mode(Smoke) takes effect");
        CHECK(scale_loops(6'000'128) == 6'000'128 / kSmokeLoopDivisor,
              "scale_loops divides by %llu in smoke mode",
              (unsigned long long)kSmokeLoopDivisor);
        CHECK(scale_loops(3) == 1, "scale_loops floors at 1");
        counter = 0;
        const BenchmarkResult r = benchmark(fn, "selftest counter (smoke)", p);
        CHECK(counter == 1, "smoke mode: %llu calls (expect 1)", (unsigned long long)counter);
        CHECK(r.min_ns_per_insn == 0.0 && r.min_clocks_per_insn == 0.0 &&
              r.cycle_source == CycleSource::Unknown,
              "smoke mode returns a zeroed result");
        CHECK(smoke_test_count() == before + 1, "smoke_test_count() incremented");
        set_run_mode(RunMode::Measure);
        CHECK(scale_loops(6'000'128) == 6'000'128, "scale_loops is identity in measure mode");
    }

    // Name filter: non-matching tests are not called at all.
    {
        set_name_filter("this-will-not-match");
        counter = 0;
        const BenchmarkResult r = benchmark(fn, "selftest counter (filtered out)", p);
        CHECK(counter == 0, "filtered-out test: %llu calls (expect 0)", (unsigned long long)counter);
        CHECK(r.total_instructions == 0, "filtered-out test returns a zeroed result");

        set_name_filter("counter");
        counter = 0;
        benchmark(fn, "selftest counter (filter match)", p);
        CHECK(counter == expected, "filter match runs normally: %llu calls",
              (unsigned long long)counter);
        set_name_filter(nullptr);
    }

    g_jit_pool->release(fn);
}

static void test_harness_measurements() {
    section("harness: chained ADD ~1 clk, linearity, throughput");

    // Register the same reference main() uses, so Tier 2 is exercised when
    // the PMU is unavailable.
    ReferenceParams ref;
    ref.fn          = gen::create_add_latency_ref(500'000, 32);
    ref.total_insns = 500'000ULL * 32;
    CHECK(ref.fn != nullptr, "compiled ADD reference function");
    set_reference_function(ref);

    BenchmarkParams p{};
    p.instructions_per_loop = 32;
    p.num_samples           = 5;
    p.num_warmup            = 2;
    p.inter_sample_ms       = 5;

    // ── Chained ADD latency: 1 cycle on every ARMv8 core ──────────────────
    {
        static constexpr uint64_t kLoops = 200'000;   // 6.4M insns, ~2 ms
        JitPool::TestFn fn = build_add_loop(kLoops, 32, 1);
        CHECK(fn != nullptr, "compiled 1-chain ADD loop");
        if (fn) {
            p.loops = kLoops;
            const BenchmarkResult r = benchmark(fn, "selftest ADD latency x32", p);
            CHECK(r.total_instructions == kLoops * 32, "total_instructions = loops x unroll");
            CHECK(r.min_ns_per_insn > 0.0, "min_ns_per_insn = %.4f > 0", r.min_ns_per_insn);
            CHECK(r.median_ns_per_insn >= r.min_ns_per_insn,
                  "median (%.4f) >= min (%.4f)", r.median_ns_per_insn, r.min_ns_per_insn);
            CHECK(std::fabs(r.min_total_ns - r.min_ns_per_insn * r.total_instructions) < 1.0,
                  "min_total_ns is consistent with min_ns_per_insn");
            CHECK(r.coeff_variation_pct >= 0.0, "CoV = %.2f%% >= 0", r.coeff_variation_pct);
            CHECK(r.cycle_source != CycleSource::Unknown,
                  "a cycle source was selected (%d)", static_cast<int>(r.cycle_source));
            CHECK_Q(r.min_clocks_per_insn > 0.80 && r.min_clocks_per_insn < 1.30,
                  "chained ADD = %.3f clk/insn (expect ~1.0)", r.min_clocks_per_insn);
            // Under the ratio tier the number is ~1 by construction (the
            // reference is the same instruction), so also check the raw wall
            // time is plausible: 1 cycle at 0.3–8 GHz is 0.125–3.3 ns.
            CHECK_Q(r.min_ns_per_insn > 0.10 && r.min_ns_per_insn < 4.0,
                  "chained ADD = %.4f ns/insn (expect 0.1-4 ns)", r.min_ns_per_insn);
            g_jit_pool->release(fn);
        }
    }

    // ── Linearity: per-instruction time is independent of unroll ──────────
    // Loop control (SUB + CBNZ) runs on an independent register and is
    // absorbed by out-of-order execution, so 8x and 64x unroll agree.
    {
        JitPool::TestFn f8  = build_add_loop(800'000, 8,  1);
        JitPool::TestFn f64 = build_add_loop(100'000, 64, 1);
        CHECK(f8 && f64, "compiled 8x and 64x unroll ADD loops");
        if (f8 && f64) {
            p.loops = 800'000; p.instructions_per_loop = 8;
            const BenchmarkResult r8  = benchmark(f8,  "selftest ADD latency x8",  p);
            p.loops = 100'000; p.instructions_per_loop = 64;
            const BenchmarkResult r64 = benchmark(f64, "selftest ADD latency x64", p);
            const double ratio = r8.min_ns_per_insn / r64.min_ns_per_insn;
            CHECK_Q(ratio > 0.85 && ratio < 1.20,
                  "ns/insn ratio 8x/64x = %.3f (expect ~1.0)", ratio);
            g_jit_pool->release(f8);
            g_jit_pool->release(f64);
        }
    }

    // ── Throughput: 4 independent chains beat the latency chain ───────────
    // Every ARM64 core arm64bench targets has at least two ALUs that can
    // execute ADD, so 4 chains must come in well under 1 clk/insn.
    {
        JitPool::TestFn fn = build_add_loop(200'000, 32, 4);
        CHECK(fn != nullptr, "compiled 4-chain ADD loop");
        if (fn) {
            p.loops = 200'000; p.instructions_per_loop = 32;
            const BenchmarkResult r = benchmark(fn, "selftest ADD tput 4 chains", p);
            CHECK_Q(r.min_clocks_per_insn > 0.0 && r.min_clocks_per_insn < 0.80,
                  "4-chain ADD = %.3f clk/insn (expect < 0.8)", r.min_clocks_per_insn);
            g_jit_pool->release(fn);
        }
    }

    set_reference_function(ReferenceParams{});
    g_jit_pool->release(ref.fn);
}

static void test_cpu_features() {
    section("CPU features (runtime detection)");

    static constexpr uint32_t kCount = static_cast<uint32_t>(CpuFeature::Count_);
    char line[512];
    size_t n = 0;
    bool stable = true;
    for (uint32_t i = 0; i < kCount; ++i) {
        const CpuFeature f = static_cast<CpuFeature>(i);
        const bool a = cpu_has(f);
        const bool b = cpu_has(f);
        if (a != b) stable = false;
        n += static_cast<size_t>(snprintf(line + n, sizeof(line) - n, "%s%s=%d",
                                          i ? " " : "", cpu_feature_name(f), a ? 1 : 0));
        if (n >= sizeof(line)) break;
    }
    note("%s", line);
    CHECK(stable, "cpu_has() is stable across repeated queries");

    // Architectural implications that hold on every real core.
    if (cpu_has(CpuFeature::LRCPC2))
        CHECK(cpu_has(CpuFeature::LRCPC), "FEAT_LRCPC2 implies FEAT_LRCPC");
    if (cpu_has(CpuFeature::I8MM))
        CHECK(cpu_has(CpuFeature::DotProd), "FEAT_I8MM implies FEAT_DotProd");
    if (cpu_has(CpuFeature::FHM))
        CHECK(cpu_has(CpuFeature::FP16), "FEAT_FHM implies FEAT_FP16");
    // Every target arm64bench supports is ARMv8.1+ with the crypto extension.
    CHECK(cpu_has(CpuFeature::LSE), "FEAT_LSE present (ARMv8.1 baseline)");
    CHECK(cpu_has(CpuFeature::AES) && cpu_has(CpuFeature::SHA256) && cpu_has(CpuFeature::CRC32),
          "AES, SHA256 and CRC32 present");
}

static void test_jit_pool_churn() {
    section("JIT pool: compile/release churn");

    static constexpr uint32_t kRounds = 2000;
    uint32_t failures = 0;
    static uint64_t counter = 0;
    for (uint32_t i = 0; i < kRounds; ++i) {
        JitPool::TestFn fn = build_counter_fn(&counter);
        if (!fn) { ++failures; continue; }
        fn();
        g_jit_pool->release(fn);
    }
    CHECK(failures == 0, "%u compile/execute/release rounds, %u failures", kRounds, failures);
    CHECK(counter == kRounds - failures, "every compiled function executed once (%llu)",
          (unsigned long long)counter);
}

// ── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            s_verbose = true;
        } else if (strcmp(argv[i], "--lenient") == 0) {
            s_lenient = true;
        } else {
            fprintf(stderr, "Usage: %s [--verbose] [--lenient]\n", argv[0]);
            return 2;
        }
    }
    if (const char* e = getenv("ARM64BENCH_SELFTEST_LENIENT"); e && *e && *e != '0')
        s_lenient = true;

#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("arm64bench selftest  (built %s %s)%s\n\n", __DATE__, __TIME__,
           s_lenient ? "  [lenient: measurement-quality checks warn only]" : "");

    JitPool jit_pool;
    g_jit_pool = &jit_pool;

    test_timer_basics();
    test_timer_monotonic();
    test_sleep_wall_time();
    test_cycle_counter();
    test_calibration();
    test_harness_call_accounting();
    test_harness_measurements();
    test_jit_pool_churn();
    test_cpu_features();

    g_jit_pool = nullptr;

    printf("\n%d passed, %d failed, %d warned, %d skipped\n", s_passed, s_failed, s_warned, s_skipped);
    printf(s_failed ? "SELFTEST FAILED\n" : "SELFTEST OK\n");
    return s_failed > 255 ? 255 : s_failed;
}
