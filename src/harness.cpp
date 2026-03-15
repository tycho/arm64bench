// harness.cpp
// Benchmark runner implementation: tick-aligned sampling, priority management,
// statistics, and formatted output.

#include "harness.h"
#include "timer.h"
#include <cstdio>
#include <cstdint>
#include <cmath>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <pthread.h>
#  include <sched.h>
#  include <time.h>
#endif

namespace arm64bench {

// ── Output mode ─────────────────────────────────────────────────────────────

static OutputMode s_output_mode = OutputMode::Console;

void set_output_mode(OutputMode mode) {
    s_output_mode = mode;
}

// ── Sleep helper ─────────────────────────────────────────────────────────────

static void sleep_ms(uint32_t ms) {
#if defined(_WIN32)
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec  = static_cast<time_t>(ms / 1000);
    ts.tv_nsec = static_cast<long>((ms % 1000) * 1'000'000L);
    nanosleep(&ts, nullptr);
#endif
}

// ── Thread priority RAII guard ───────────────────────────────────────────────
//
// Elevating priority reduces the probability that the OS preempts us mid-
// sample. It does not eliminate it — the OS can still preempt real-time
// threads for interrupt service — but it substantially reduces the rate of
// outlier samples.
//
// Failures to elevate are silently ignored: the benchmark still runs, just
// with higher noise (reflected in CoV). This is preferable to hard-failing
// for unprivileged users.

struct PriorityGuard {
#if defined(_WIN32)
    int     old_thread_priority;
    DWORD   old_process_priority_class;

    PriorityGuard() {
        old_thread_priority       = GetThreadPriority(GetCurrentThread());
        old_process_priority_class = GetPriorityClass(GetCurrentProcess());
        SetPriorityClass(GetCurrentProcess(),  HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(),  THREAD_PRIORITY_HIGHEST);
    }

    ~PriorityGuard() {
        SetThreadPriority(GetCurrentThread(),  old_thread_priority);
        SetPriorityClass(GetCurrentProcess(),  old_process_priority_class);
    }

#else // POSIX

    struct sched_param  old_param;
    int                 old_policy;
    bool                elevated;

    PriorityGuard() : elevated(false) {
        pthread_getschedparam(pthread_self(), &old_policy, &old_param);

        // SCHED_FIFO requires CAP_SYS_NICE on Linux, or running as root.
        // On macOS it requires root or specific entitlements.
        // Failure here is expected for normal users and is benign.
        struct sched_param p{};
        p.sched_priority = sched_get_priority_max(SCHED_FIFO);
        elevated = (pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) == 0);

#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    }

    ~PriorityGuard() {
        if (elevated)
            pthread_setschedparam(pthread_self(), old_policy, &old_param);
    }
#endif
};

// ── Statistics ───────────────────────────────────────────────────────────────

struct Stats {
    double min;
    double median;
    double mean;
    double coeff_variation_pct;
};

// Sorts samples[] ascending in-place (insertion sort; N ≤ 32 always).
// Computes statistics over the first (count - discard_highest) elements
// (i.e., the fastest retained samples).
static Stats compute_stats(double* samples, uint32_t count, uint32_t discard_highest) {
    // Insertion sort: optimal for N ≤ ~20, acceptable for N ≤ 32.
    for (uint32_t i = 1; i < count; ++i) {
        const double key = samples[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && samples[j] > key) {
            samples[j + 1] = samples[j];
            --j;
        }
        samples[j + 1] = key;
    }

    // After sorting, samples[0] is fastest (smallest), samples[count-1] is
    // slowest (most likely an OS preemption). We retain the fastest
    // (count - discard_highest) samples.
    const uint32_t retained = count - discard_highest;

    const double min = samples[0];

    // Median of retained samples.
    double median;
    if (retained % 2 == 1) {
        median = samples[retained / 2];
    } else {
        median = (samples[retained / 2 - 1] + samples[retained / 2]) * 0.5;
    }

    // Population mean and standard deviation over retained samples.
    double sum = 0.0;
    for (uint32_t i = 0; i < retained; ++i)
        sum += samples[i];
    const double mean = sum / static_cast<double>(retained);

    double var_acc = 0.0;
    for (uint32_t i = 0; i < retained; ++i) {
        const double d = samples[i] - mean;
        var_acc += d * d;
    }
    // Use population std dev (divide by N, not N-1): we're characterising
    // measurement noise, not estimating a population parameter.
    const double std_dev = (retained > 1)
        ? std::sqrt(var_acc / static_cast<double>(retained))
        : 0.0;

    const double coeff_variation_pct = (mean > 0.0)
        ? (std_dev / mean * 100.0)
        : 0.0;

    return { min, median, mean, coeff_variation_pct };
}

// ── Output formatting ────────────────────────────────────────────────────────

void print_csv_header() {
    printf("name,"
           "min_ns_per_insn,median_ns_per_insn,"
           "min_clocks_per_insn,"
           "coeff_variation_pct,noisy,"
           "total_instructions,bandwidth_gbs\n");
}

static void print_result(const char* name, const BenchmarkResult& r) {
    if (s_output_mode == OutputMode::CSV) {
        printf("%s,%.4f,%.4f,%.4f,%.2f,%d,%llu,%.2f\n",
               name,
               r.min_ns_per_insn,
               r.median_ns_per_insn,
               r.min_clocks_per_insn,
               r.coeff_variation_pct,
               r.noisy ? 1 : 0,
               static_cast<unsigned long long>(r.total_instructions),
               r.bandwidth_gbs);
    } else {
        // Console: fixed-width columns designed to align across a typical run.
        //
        // Example:
        //   ADD (chained) x32          : min   0.305 ns  med   0.307 ns   1.220 clk  CoV  0.3%
        //   ADD (indep)   x32          : min   0.038 ns  med   0.040 ns   0.153 clk  CoV  0.8%
        //   SDIV (chained) x8          : min   8.127 ns  med   8.201 ns  32.508 clk  CoV  1.1%

        printf("%-36s: min %7.3f ns  med %7.3f ns",
               name,
               r.min_ns_per_insn,
               r.median_ns_per_insn);

        if (r.min_clocks_per_insn > 0.0)
            printf("  %7.3f clk", r.min_clocks_per_insn);
        else
            printf("  ??? .??? clk");

        printf("  CoV %4.1f%%%s",
               r.coeff_variation_pct,
               r.noisy ? "  !" : "   ");

        if (r.bandwidth_gbs > 0.0)
            printf("  →%7.1f GB/s", r.bandwidth_gbs);

        printf("\n");
    }

    fflush(stdout);
}

// ── Benchmark runner ─────────────────────────────────────────────────────────

BenchmarkResult benchmark(TestFn fn, const char* name, const BenchmarkParams& params) {
    // Hard cap so the sample array stays on the stack.
    static constexpr uint32_t kMaxSamples = 32;

    const uint32_t num_samples = (params.num_samples < kMaxSamples)
                               ? params.num_samples : kMaxSamples;

    // discard_highest must leave at least 1 retained sample.
    const uint32_t discard_highest = (params.discard_highest < num_samples)
                                   ? params.discard_highest : 0;

    const uint64_t total_insns =
        static_cast<uint64_t>(params.loops) * params.instructions_per_loop;

    const double inv_insns = 1.0 / static_cast<double>(total_insns);

    // ── Warm-up ───────────────────────────────────────────────────────────
    // Untimed calls to bring the I-cache, D-cache, and branch predictor into
    // a steady state before we start measuring. Without this, the first timed
    // sample pays for cold I-cache misses on the JIT'd code, TLB fills for
    // any data buffers, and branch predictor cold start — all of which are not
    // what we're trying to measure.
    for (uint32_t i = 0; i < params.num_warmup; ++i)
        fn();

    // Brief sleep after warm-up: the warm-up loop raises the CPU's power
    // demand, which may briefly boost the frequency above its steady state.
    // A short sleep lets the governor settle before we start measuring.
    sleep_ms(params.inter_sample_ms);

    double sample_ns[kMaxSamples];

    // ── Timed sampling (priority-elevated) ───────────────────────────────
    {
        PriorityGuard priority_guard;

        for (uint32_t s = 0; s < num_samples; ++s) {
            // Align to a timer tick boundary. wait_for_tick() returns the
            // counter value at the moment the tick advanced, so t0 is exact.
            // The overhead between wait_for_tick() returning and fn() being
            // called is a single indirect branch — negligible relative to
            // the millions of instructions fn() will execute.
            const RawTick t0 = wait_for_tick();
            fn();
            const RawTick t1 = tick_now();

            sample_ns[s] = ticks_to_ns_f(t1 - t0);

            // Sleep between samples (but not after the last one — no point
            // paying the sleep cost when we're done collecting).
            if (s + 1 < num_samples)
                sleep_ms(params.inter_sample_ms);
        }

        // PriorityGuard destructor restores normal priority here.
    }

    // ── Statistics ────────────────────────────────────────────────────────
    // compute_stats sorts sample_ns in-place; that's fine since we own it.
    const Stats stats = compute_stats(sample_ns, num_samples, discard_highest);

    BenchmarkResult result{};
    result.total_instructions  = total_insns;
    result.min_total_ns        = stats.min;
    result.min_ns_per_insn     = stats.min    * inv_insns;
    result.median_ns_per_insn  = stats.median * inv_insns;
    result.coeff_variation_pct = stats.coeff_variation_pct;
    result.noisy               = (stats.coeff_variation_pct > params.noise_threshold_pct);

    if (g_cpu_freq_hz > 0) {
        // clk/insn = (ns/insn) * 1e-9 * Hz_cpu
        result.min_clocks_per_insn =
            result.min_ns_per_insn * 1e-9 * static_cast<double>(g_cpu_freq_hz);
    }

    if (params.bytes_per_insn > 0 && result.min_ns_per_insn > 0.0) {
        // bandwidth (GB/s) = bytes / ns  (units cancel: B/ns = B*1e9/s / 1e9 = GB/s)
        result.bandwidth_gbs =
            static_cast<double>(params.bytes_per_insn) / result.min_ns_per_insn;
    }

    print_result(name, result);
    return result;
}

} // namespace arm64bench
