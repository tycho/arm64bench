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

// ── Cycle source ────────────────────────────────────────────────────────────

// Indicates how min_clocks_per_insn was derived, in decreasing reliability order.
enum class CycleSource : uint8_t {
    Unknown    = 0, // frequency unknown, ratio not available
    Calibrated = 1, // wall-clock × calibrated frequency (Tier 3; may drift)
    Ratio      = 2, // ratio vs. 1-cycle reference sandwich (Tier 2; drift-resistant)
    PMU        = 3, // hardware PMU cycle counter (Tier 1; P-state immune)
};

// ── Signature for JIT-generated test functions ──────────────────────────────

// No arguments, no return value. All parameters (loop count, register setup, etc.)
// are baked in at JIT compile time. Called through a plain indirect branch.
using TestFn = void (*)();

// ── Reference function for Tier 2 ratio normalization ───────────────────────

// A registered reference function is run (untimed) once to warm I-cache, then
// timed once before and once after every timed test sample. The ratio
//
//   ratio = (test_ns / test_total_insns) / avg(ref_before_ns, ref_after_ns)
//            ────────────────────────────   ─────────────────────────────────
//               test ns/insn                  ref ns/insn  (≈ 1 cycle worth)
//
// cancels out the clock frequency, making CPI portable across machines and
// robust against external P-state drift between samples.
//
// The reference should be a chained 1-cycle-per-instruction loop (e.g. ADD
// latency chain) so that ratio directly equals the test's CPI.
//
// Instability detection: if |ref_after - ref_before| / ref_before exceeds
// instability_pct, the CPU clock changed between the two reference probes
// (external throttle). The sample is discarded and re-taken up to retry_limit
// times. Note: this does NOT catch instruction-induced throttling (where the
// test instructions themselves change the clock; both reference probes see the
// un-throttled rate). For SVE2 wide ops, use PMU counters instead.
struct ReferenceParams {
    TestFn   fn              = nullptr;
    uint64_t total_insns     = 0;     // loops × insns_per_loop baked into fn
    double   instability_pct = 2.0;   // % divergence threshold to flag/retry
    uint32_t retry_limit     = 3;     // max retries per sample on instability
};

// Register a global reference function for Tier 2 ratio normalization.
// Call once after JIT pool initialisation. Pass a default-constructed
// ReferenceParams{} (fn == nullptr) to disable.
void set_reference_function(const ReferenceParams& ref);

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

    // Untimed calls to the test function immediately before each timed
    // sample (in addition to the global num_warmup at session start).
    // Re-primes the L1 I/D-cache after any thread migration that occurred
    // during the preceding inter-sample sleep. 1 is sufficient for most
    // tests; 0 disables the per-sample warmup.
    uint32_t    num_per_sample_warmup = 1;

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

    // Best available cycles-per-instruction estimate.
    // See cycle_source for which tier was used.
    // 0.0 when CycleSource::Unknown.
    double      min_clocks_per_insn;
    CycleSource cycle_source;

    // Ratio normalization details (valid when cycle_source == CycleSource::Ratio).
    // ratio_unstable is set when the reference diverged on the final attempt
    // (retry limit exhausted), indicating potential residual P-state noise.
    bool        ratio_unstable;

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

// Run the benchmark described by params:
//   1. num_warmup untimed calls to fn (cache/predictor warm-up).
//   2. Brief sleep to let the CPU frequency settle after warm-up load.
//   3. Elevate thread priority for the duration of timed sampling.
//   4. For each of num_samples:
//        a. num_per_sample_warmup untimed calls (re-prime L1 after migration).
//        b. If reference function registered: timed reference before (Tier 2).
//        c. Call wait_for_tick(); read PMU cycle counter if available (Tier 1).
//        d. Call fn() and record elapsed ticks and cycles.
//        e. If reference function registered: timed reference after; check
//           stability; retry up to retry_limit times if unstable.
//   5. Drop priority, sort samples, discard the discard_highest slowest.
//   6. Compute and print statistics. Return BenchmarkResult.
//      min_clocks_per_insn uses the best available source: PMU > Ratio > Calibrated.
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
