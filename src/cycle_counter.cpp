// cycle_counter.cpp
// Hardware PMU cycle counter implementation.
//
// ── macOS (Apple Silicon) ──────────────────────────────────────────────────
//
// The kpc framework in /usr/lib/system/libsystem_kernel.dylib exposes
// per-thread hardware performance counter access. We load it at runtime via
// dlsym to avoid a hard link dependency and to allow graceful fallback when
// the symbols are not present.
//
// Counter layout (Apple Silicon M-series, KPC_CLASS_FIXED):
//   counter[0] = CPU cycles elapsed on the calling thread
//   counter[1] = Instructions retired on the calling thread
//
// kpc_force_all_ctrs_set(1) allocates all PMU counters to userspace; it
// may fail without root, in which case fixed counters still often work
// because they are always-running. We attempt the call and continue.
//
// The sanity check (a timed busy loop) verifies that counter[0] actually
// advances at a CPU-cycle rate before marking the counter as available.
// This catches the case where kpc calls succeed but return zeros or garbage.
//
// ── Windows ARM64 (Qualcomm Snapdragon / Oryon) ────────────────────────────
//
// On Qualcomm Snapdragon systems, the PMU driver (qpmu.sys) typically sets
// PMUSERENR_EL0.EN = 1, allowing EL0 userspace to read PMCCNTR_EL0 directly
// without a kernel trap. We probe by attempting a read inside a Structured
// Exception Handler (__try/__except). If access is denied (PMUSERENR_EL0.EN
// not set), the MRS is trapped by EL1 and Windows delivers
// EXCEPTION_ILLEGAL_INSTRUCTION; the handler catches it and we fall back to
// Tier 2 ratio normalization.
//
// Note: __rdtsc() on Windows ARM64 maps to CNTVCT_EL0 (the fixed-frequency
// generic timer, ~24 MHz on Snapdragon), NOT PMCCNTR_EL0 (CPU cycles). It is
// not suitable for cycle counting and is not used here.
//
// ── Linux ──────────────────────────────────────────────────────────────────
//
// Not yet implemented. perf_event_open(PERF_COUNT_HW_CPU_CYCLES) per-thread
// is the right path.

#include "cycle_counter.h"
#include "timer.h"

#ifdef __APPLE__
#include <dlfcn.h>
#include <cstdio>

// ── kpc constants ──────────────────────────────────────────────────────────

static constexpr uint32_t KPC_CLASS_FIXED      = 0;
static constexpr uint32_t KPC_CLASS_FIXED_MASK = (1u << KPC_CLASS_FIXED); // 1
static constexpr uint32_t KPC_MAX_COUNTERS     = 32;

// ── kpc function pointer types ─────────────────────────────────────────────

// kpc_force_all_ctrs_set: allocate all PMU counters to userspace.
// May require root; failure is non-fatal for fixed counters.
typedef int      (*kpc_force_all_ctrs_set_t)(int val);

// kpc_set_counting: choose which counter classes to enable system-wide.
typedef int      (*kpc_set_counting_t)(uint32_t classes);

// kpc_set_thread_counting: enable per-thread accumulation for given classes.
typedef int      (*kpc_set_thread_counting_t)(uint32_t classes);

// kpc_get_counter_count: number of counters in the given class mask.
typedef uint32_t (*kpc_get_counter_count_t)(uint32_t classes);

// kpc_get_thread_counters: read per-thread counter values into buf.
//   tid = 0 → calling thread.
//   buf_count = number of elements in buf (use kpc_get_counter_count result).
typedef int      (*kpc_get_thread_counters_t)(int tid, unsigned int buf_count, uint64_t* buf);

// ── Module state ───────────────────────────────────────────────────────────

static kpc_force_all_ctrs_set_t  s_force_ctrs   = nullptr;
static kpc_set_counting_t        s_set_counting  = nullptr;
static kpc_set_thread_counting_t s_set_thread    = nullptr;
static kpc_get_counter_count_t   s_get_count     = nullptr;
static kpc_get_thread_counters_t s_get_thread    = nullptr;

static bool     s_available     = false;
static bool     s_init_attempted = false;
static uint32_t s_counter_count = 0;

// Read buffer: reused across calls to avoid stack allocation on the hot path.
// Access is single-threaded (the benchmark runner uses one thread).
static uint64_t s_read_buf[KPC_MAX_COUNTERS];

namespace arm64bench {

bool cycle_counter_init() {
    if (s_init_attempted) return s_available;
    s_init_attempted = true;

    // kpc symbol locations vary by macOS version:
    //   macOS ≤ 14 (Sonoma):  /usr/lib/system/libsystem_kernel.dylib
    //   macOS ≥ 15 (Sequoia): /System/Library/PrivateFrameworks/kperf.framework/kperf
    //
    // We try the known paths in order, then fall back to RTLD_DEFAULT (which
    // succeeds if any path above already loaded the containing library).
    // Loading through the dyld shared cache works even when the on-disk path
    // is a broken symlink (the framework binary lives in the cache).
    static const char* const kCandidates[] = {
        "/System/Library/PrivateFrameworks/kperf.framework/kperf",
        "/usr/lib/system/libsystem_kernel.dylib",
    };

    void* lib = nullptr;
    for (const char* path : kCandidates) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        // Verify at least one kpc symbol is present before committing.
        if (dlsym(h, "kpc_set_counting")) { lib = h; break; }
    }
    if (!lib) lib = RTLD_DEFAULT;   // already loaded by a dependency?

    s_force_ctrs   = (kpc_force_all_ctrs_set_t)  dlsym(lib, "kpc_force_all_ctrs_set");
    s_set_counting = (kpc_set_counting_t)         dlsym(lib, "kpc_set_counting");
    s_set_thread   = (kpc_set_thread_counting_t)  dlsym(lib, "kpc_set_thread_counting");
    s_get_count    = (kpc_get_counter_count_t)    dlsym(lib, "kpc_get_counter_count");
    s_get_thread   = (kpc_get_thread_counters_t)  dlsym(lib, "kpc_get_thread_counters");

    if (!s_set_counting || !s_set_thread || !s_get_count || !s_get_thread)
        return false;

    // Get the number of fixed counters first — if this returns 0 the
    // hardware or kernel doesn't support what we need.
    s_counter_count = s_get_count(KPC_CLASS_FIXED_MASK);
    if (s_counter_count == 0 || s_counter_count > KPC_MAX_COUNTERS) return false;

    // Attempt to allocate all counters to userspace and enable system-wide
    // counting. These calls may require root (EPERM without it); failure is
    // non-fatal because fixed counters on Apple Silicon are always-running
    // hardware counters — they tick whether or not we requested them.
    if (s_force_ctrs) s_force_ctrs(1);        // EPERM without root: non-fatal
    s_set_counting(KPC_CLASS_FIXED_MASK);     // may fail: non-fatal

    // Enable per-thread accumulation. On macOS ≤ 14 this succeeded without
    // root for fixed counters; macOS ≥ 15 (Sequoia/Tahoe) requires root or
    // a private entitlement for all kpc calls, so this may return EPERM(-1).
    if (s_set_thread(KPC_CLASS_FIXED_MASK) != 0) return false;

    // ── Sanity check ──────────────────────────────────────────────────────
    // Verify that counter[0] advances at a CPU-cycle rate. We run a busy
    // loop for a measured wall-clock interval and check that the cycle delta
    // is consistent with the known fixed-timer frequency (24 MHz on Apple
    // Silicon → 1 ns per tick since tick_now() returns nanoseconds on POSIX).
    uint64_t c_before[KPC_MAX_COUNTERS] = {};
    uint64_t c_after[KPC_MAX_COUNTERS]  = {};

    if (s_get_thread(0, s_counter_count, c_before) != 0) return false;

    // Busy-loop for approximately 1ms of wall time.
    const RawTick wall_start = tick_now();
    volatile uint64_t v = 1;
    while (tick_now() - wall_start < 1'000'000ULL) // 1ms in ns
        v ^= v * 6364136223846793005ULL + 1442695040888963407ULL;
    (void)v;
    const RawTick wall_elapsed_ns = tick_now() - wall_start;

    if (s_get_thread(0, s_counter_count, c_after) != 0) return false;

    const uint64_t cycle_delta = c_after[0] - c_before[0];

    // Derive an implied frequency from the measured delta.
    // Expected: ~500MHz–8GHz for any real ARM64 device at any P-state.
    // cycles / (elapsed_ns * 1e-9) = cycles_per_second = Hz
    const double implied_ghz = (wall_elapsed_ns > 0)
        ? (static_cast<double>(cycle_delta) / static_cast<double>(wall_elapsed_ns))
        : 0.0;

    // Accept 0.2 GHz–20 GHz (very generous; catches stuck-at-zero counters
    // and any wildly wrong counter index).
    if (implied_ghz < 0.2 || implied_ghz > 20.0) return false;

    s_available = true;
    return true;
}

bool cycle_counter_available() {
    return s_available;
}

uint64_t cycle_counter_read() {
    if (!s_available) return 0;
    s_get_thread(0, s_counter_count, s_read_buf);
    return s_read_buf[0];
}

} // namespace arm64bench

#elif defined(_WIN32) // ── Windows ARM64 ────────────────────────────────────

// __builtin_arm_rsr64 is a Clang built-in for reading ARM64 system registers
// by name. arm64bench requires Clang (MSVC is rejected), so this is safe.
// It compiles to a single MRS instruction; no constant encoding required.

#include <windows.h>

namespace arm64bench {

static bool s_available      = false;
static bool s_init_attempted = false;

// Probe PMCCNTR_EL0 inside a minimal SEH frame. If PMUSERENR_EL0.EN is not
// set, the MRS is trapped by EL1 and Windows raises an exception here.
// Using EXCEPTION_EXECUTE_HANDLER catches any exception code (the specific
// code may be EXCEPTION_ILLEGAL_INSTRUCTION or EXCEPTION_PRIV_INSTRUCTION
// depending on the Qualcomm driver version).
static bool probe_pmccntr(uint64_t* out) {
    __try {
        *out = __builtin_arm_rsr64("PMCCNTR_EL0");
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool cycle_counter_init() {
    if (s_init_attempted)
        return s_available;
    s_init_attempted = true;

    // Probe access; bail out if EL0 reads are not permitted.
    uint64_t c_before = 0;
    if (!probe_pmccntr(&c_before))
        return false;

    // PMCCNTR_EL0 is a per-CPU register: each physical core has its own
    // counter. If the thread migrates between reads, we get values from two
    // different counters — the delta can be billions of cycles and makes the
    // implied frequency absurdly high. Pin to the current CPU for the duration
    // of this sanity check to prevent that.
    const DWORD cpu = GetCurrentProcessorNumber();
    const DWORD_PTR old_mask = SetThreadAffinityMask(
        GetCurrentThread(), DWORD_PTR(1) << cpu);

    // Sanity check: verify the counter advances at a plausible CPU-cycle rate.
    // On Windows, RawTick is QPC ticks (not ns), so use tick_frequency() to
    // compute a 1ms spin duration and ticks_to_ns_f() for the ns denominator.
    const uint64_t one_ms_ticks = tick_frequency() / 1000;

    if (!probe_pmccntr(&c_before)) {  // re-read after affinity change settles
        if (old_mask)
            SetThreadAffinityMask(GetCurrentThread(), old_mask);
        return false;
    }

    const RawTick wall_start = tick_now();
    volatile uint64_t v = 1;
    while (tick_now() - wall_start < one_ms_ticks)
        v ^= v * 6364136223846793005ULL + 1442695040888963407ULL;
    (void)v;
    const double wall_elapsed_ns = ticks_to_ns_f(tick_now() - wall_start);

    uint64_t c_after = 0;
    if (!probe_pmccntr(&c_after)) {
        if (old_mask)
            SetThreadAffinityMask(GetCurrentThread(), old_mask);
        return false;
    }

    if (old_mask) SetThreadAffinityMask(GetCurrentThread(), old_mask);

    const uint64_t cycle_delta = c_after - c_before;

    // implied_ghz = cycles / elapsed_ns  (since GHz = cycles / ns)
    const double implied_ghz = (wall_elapsed_ns > 0.0)
        ? (static_cast<double>(cycle_delta) / wall_elapsed_ns)
        : 0.0;

    // Accept 0.2–20 GHz; rejects stuck-at-zero counters and implausible values.
    if (implied_ghz < 0.2 || implied_ghz > 20.0)
        return false;

    s_available = true;
    return true;
}

bool cycle_counter_available() {
    return s_available;
}

// Hot path: read directly without SEH. init() already verified access works.
uint64_t cycle_counter_read() {
    if (!s_available)
        return 0;
    return __builtin_arm_rsr64("PMCCNTR_EL0");
}

} // namespace arm64bench

#else // ── Non-Apple, non-Windows stub (Linux etc.) ──────────────────────────

// TODO: Linux — perf_event_open(PERF_COUNT_HW_CPU_CYCLES) per-thread.

namespace arm64bench {

bool     cycle_counter_init()      { return false; }
bool     cycle_counter_available() { return false; }
uint64_t cycle_counter_read()      { return 0; }

} // namespace arm64bench

#endif // __APPLE__ / _WIN32
