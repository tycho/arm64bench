#pragma once
// harness.h
// Benchmark runner: tick-aligned sampling, statistics, and output.
//
// Usage pattern:
//
//   BenchmarkParams p;
//   p.loops               = 6'000'128;   // baked into the JIT'd function
//   p.instructions_per_loop = 32;         // likewise
//
//   arm64bench::benchmark(my_jit_fn, "ADD (chained, 32x unroll)", p);
//
// The harness does NOT generate or free test functions; that is the
// responsibility of the JIT layer. The harness only times them.

#include "timer.h"    // RawTick, tick_now, wait_for_tick
#include <cstdint>

namespace arm64bench {

// ── Parameters ─────────────────────────────────────────────────────────────

struct BenchmarkParams {
    // --- What the JIT'd function does (for normalization only) ---
    //
    // Both values are baked into the generated machine code at compile time.
    // The harness uses them only to divide raw elapsed time by total
    // instruction count, producing a per-instruction result.
    uint64_t    loops;                  // iteration count of the JIT loop
    uint32_t    instructions_per_loop;  // instructions per iteration

    // --- Sampling strategy ---

    // Number of timed samples to collect.
    uint32_t    num_samples         = 7;

    // Untimed warm-up calls before timed sampling begins.
    // Ensures I-cache is warm, branch predictors are trained, and any
    // data buffers touched by the test are resident in L1/L2.
    uint32_t    num_warmup          = 2;

    // Number of slowest samples to discard before computing statistics.
    // These represent OS scheduler preemptions, thermal throttle events,
    // P→E core migrations, etc. The minimum of the retained samples is
    // the best approximation of the true hardware throughput.
    uint32_t    discard_highest     = 1;

    // Milliseconds to sleep between timed samples.
    // Gives the OS scheduler time to service other threads, and gives the
    // CPU frequency governor time to re-stabilize after the burst of work.
    uint32_t    inter_sample_ms     = 20;

    // Results whose coefficient of variation (%) exceeds this threshold
    // are flagged as noisy in the output. High CoV suggests OS interference,
    // frequency scaling mid-run, or a test that is right at a cache boundary.
    double      noise_threshold_pct = 5.0;

    // Optional: bytes of memory traffic per "instruction" (harness normalization
    // unit). When non-zero, the harness computes and displays memory bandwidth.
    //
    // For memory tests, set instructions_per_loop = cache_lines_per_pass and
    // bytes_per_insn = 64 (one cache line). Then:
    //   bandwidth (GB/s) = bytes_per_insn / min_ns_per_insn
    // since ns/cache_line × 10^-9 s/ns gives s/cache_line, and
    // bytes / (s/cache_line) / 10^9 = bytes * 10^9 / (ns * 10^9) = bytes / ns.
    uint32_t    bytes_per_insn      = 0;
};

// ── Results ─────────────────────────────────────────────────────────────────

struct BenchmarkResult {
    // Per-instruction timings in nanoseconds, computed from the minimum
    // and median of the retained (post-discard) samples.
    double      min_ns_per_insn;
    double      median_ns_per_insn;

    // Coefficient of variation (std_dev / mean * 100) across retained samples.
    // Low CoV = stable, reproducible measurement.
    // High CoV = noisy; consider increasing num_samples or num_warmup.
    double      coeff_variation_pct;
    bool        noisy;              // true iff CoV% > noise_threshold_pct

    // Estimated clock cycles per instruction. 0.0 if g_cpu_freq_hz is not set.
    // Derived from min_ns_per_insn and the calibrated CPU frequency.
    double      min_clocks_per_insn;

    // Raw minimum elapsed nanoseconds for a complete call to the test function.
    // Useful for sanity-checking: should be roughly loops * instructions_per_loop
    // * min_ns_per_insn (by definition).
    double      min_total_ns;

    // Total instructions per call: loops * instructions_per_loop.
    uint64_t    total_instructions;

    // Memory bandwidth in GB/s. Non-zero only when BenchmarkParams::bytes_per_insn
    // was set. Derived from min_ns_per_insn and bytes_per_insn.
    double      bandwidth_gbs;
};

// ── Interface ───────────────────────────────────────────────────────────────

// Signature of a JIT-generated (or static inline asm) test function.
// No arguments, no return value. All parameters (loop count, register
// setup, etc.) are baked in at JIT compile time. The harness calls this
// through a plain indirect branch with no special ABI requirements beyond
// the platform's standard C calling convention.
using TestFn = void (*)();

// Run the benchmark described by params:
//   1. num_warmup untimed calls to fn (cache/predictor warm-up).
//   2. Brief sleep to let the CPU frequency settle after warm-up load.
//   3. Elevate thread priority for the duration of timed sampling.
//   4. For each of num_samples:
//        a. Call wait_for_tick() to align to a timer boundary.
//        b. Call fn() and record elapsed ticks.
//        c. Sleep inter_sample_ms before the next sample.
//   5. Drop priority, sort samples, discard the discard_highest slowest.
//   6. Compute and print statistics. Return BenchmarkResult.
BenchmarkResult benchmark(TestFn fn, const char* name, const BenchmarkParams& params);

// ── Output control ──────────────────────────────────────────────────────────

enum class OutputMode {
    Console,    // human-readable fixed-width table (default)
    CSV,        // machine-readable, one row per benchmark
};

void set_output_mode(OutputMode mode);

// When OutputMode::CSV is active, call this once before the first benchmark()
// to emit the header row.
void print_csv_header();

} // namespace arm64bench
