#pragma once
// timer.h
// Platform-abstracted high-resolution monotonic timer for arm64bench.
//
// Key design points:
//
//  RawTick is an opaque counter type. On POSIX it is nanoseconds from
//  clock_gettime(CLOCK_MONOTONIC); on Windows it is QPC ticks. Either way,
//  you should treat it as an opaque 64-bit integer and only interpret
//  differences via the conversion functions below.
//
//  On ARM64 the hardware generic timer (CNTVCT_EL0) typically runs at 24MHz
//  (period ~41.7ns) or 19.2MHz. This is the *timer* frequency, not the CPU
//  frequency. g_cpu_freq_hz is a separate, calibrated value for expressing
//  results in clock cycles.

#include <cstdint>

namespace arm64bench {

// Opaque counter from the platform's best monotonic timer.
//   POSIX:   nanoseconds from clock_gettime(CLOCK_MONOTONIC).
//   Windows: QueryPerformanceCounter ticks.
using RawTick = uint64_t;

// Returns the current raw counter value. As fast as the platform allows.
RawTick tick_now();

// Returns the counter's frequency in Hz (ticks per second).
//   POSIX:   always 1,000,000,000 (ticks are nanoseconds).
//   Windows: QPC frequency, typically 10,000,000 on modern hardware.
// Result is cached after the first call.
uint64_t tick_frequency();

// Convert a raw tick *delta* to whole nanoseconds (rounds down).
uint64_t ticks_to_ns(RawTick delta);

// Convert a raw tick *delta* to nanoseconds as double (full precision).
// Prefer this for per-instruction calculations where fractional ns matters.
double ticks_to_ns_f(RawTick delta);

// Spin until the underlying counter visibly advances past its current value,
// then return the new counter value.
//
// Use this immediately before the start of a timed region. Rationale:
// Even on POSIX where we read nanoseconds, the actual timer advances in
// discrete steps (e.g. ~41ns at 24MHz, or ~100ns at 10MHz on Windows QPC).
// If we start measuring in the middle of a step, the first partial step
// inflates t0 and deflates the measured elapsed time by up to one step.
// Aligning to a step boundary eliminates this bias at the cost of a brief
// spin (at most one step's worth of time).
RawTick wait_for_tick();

// Estimated CPU core frequency in Hz. Zero until set by calibration.
//
// This is entirely separate from tick_frequency(). The timer counter on ARM64
// runs at a fixed 24MHz regardless of CPU P-state or boost state. To convert
// timing measurements to clock cycles we need the actual CPU frequency, which
// must be measured empirically (no kernel-readable MSR equivalent exists on
// ARM64 Windows/macOS without elevated privileges).
//
// Set this via calibrate_cpu_freq() or via the --MHz command-line option.
extern uint64_t g_cpu_freq_hz;

// Convert a raw tick delta to estimated CPU clock cycles.
// Returns 0.0 if g_cpu_freq_hz has not been set.
double ticks_to_cycles(RawTick delta);

// Run a calibration pass to estimate g_cpu_freq_hz.
// Executes a known-latency instruction loop, times it with the wall clock,
// and sets g_cpu_freq_hz from the result. Should be called once at startup
// before any benchmarks run.
//
// Returns the estimated frequency in Hz, and also writes it to g_cpu_freq_hz.
// Returns 0 if calibration fails or produces an implausible result.
uint64_t calibrate_cpu_freq();

} // namespace arm64bench
