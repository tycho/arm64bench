// timer.cpp
// Platform-specific high-resolution timer implementation.

#include "timer.h"
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#  include <time.h>
#else
#  error "Unsupported platform: expected Windows, macOS, or Linux"
#endif

namespace arm64bench {

uint64_t g_cpu_freq_hz = 0;

// ── POSIX (macOS + Linux) ──────────────────────────────────────────────────

#if !defined(_WIN32)

RawTick tick_now() {
    struct timespec ts;
    // CLOCK_MONOTONIC: guaranteed non-decreasing, unaffected by wall-clock
    // adjustments (NTP steps, daylight saving, etc.).
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec)  * 1'000'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t tick_frequency() {
    // On POSIX the raw tick is nanoseconds, so frequency is 1GHz.
    return 1'000'000'000ULL;
}

uint64_t ticks_to_ns(RawTick delta) {
    // Ticks are already nanoseconds on POSIX.
    return delta;
}

double ticks_to_ns_f(RawTick delta) {
    return static_cast<double>(delta);
}

// ── Windows ────────────────────────────────────────────────────────────────

#else // _WIN32

static uint64_t s_qpf = 0;

// Retrieve and cache the QPC frequency. Thread-safe under the assumption
// that this is called before multiple threads might race on it (i.e. call
// tick_frequency() from main() before spawning worker threads).
static uint64_t get_qpf() {
    if (!s_qpf) {
        LARGE_INTEGER li;
        QueryPerformanceFrequency(&li);
        s_qpf = static_cast<uint64_t>(li.QuadPart);
    }
    return s_qpf;
}

RawTick tick_now() {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

uint64_t tick_frequency() {
    return get_qpf();
}

uint64_t ticks_to_ns(RawTick delta) {
    // Naively computing (delta * 1'000'000'000) / freq overflows uint64_t when
    // delta is large (e.g. after ~9 seconds at 1GHz QPC). Split the division:
    //
    //   ns = floor(delta / freq) * 1e9 + floor((delta % freq) * 1e9 / freq)
    //
    // The second term: (delta % freq) < freq ≤ 10MHz, and 10MHz * 1e9 = 1e16,
    // which fits in uint64_t (max ~1.8e19).
    const uint64_t freq = get_qpf();
    return (delta / freq) * 1'000'000'000ULL
         + (delta % freq) * 1'000'000'000ULL / freq;
}

double ticks_to_ns_f(RawTick delta) {
    // Floating-point: no overflow risk, but loses sub-nanosecond precision for
    // very large deltas. Acceptable for per-instruction calculations.
    return static_cast<double>(delta) * (1e9 / static_cast<double>(get_qpf()));
}

#endif // _WIN32

// ── Shared ─────────────────────────────────────────────────────────────────

RawTick wait_for_tick() {
    // Spin-read until the counter changes. On POSIX the counter is nanoseconds
    // and the kernel typically advances it in larger discrete steps (e.g. ~42ns
    // at 24MHz). On Windows QPC the step is ~100ns at 10MHz. Either way, the
    // spin terminates within one step period (sub-microsecond).
    const RawTick start = tick_now();
    RawTick now;
    do {
        now = tick_now();
    } while (now == start);
    return now;
}

double ticks_to_cycles(RawTick delta) {
    if (!g_cpu_freq_hz)
        return 0.0;
    // cycles = seconds * Hz = (nanoseconds * 1e-9) * Hz
    return ticks_to_ns_f(delta) * 1e-9 * static_cast<double>(g_cpu_freq_hz);
}

// calibrate_cpu_freq() is implemented in calibrate.cpp, which uses the JIT
// pool to emit a known-latency instruction loop and times it against the
// wall clock. It is declared in timer.h and lives in a separate translation
// unit to avoid a circular dependency: timer.cpp must not include jit_buffer.h.

} // namespace arm64bench
