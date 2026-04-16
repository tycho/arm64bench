#pragma once
// cycle_counter.h
// Hardware PMU cycle counter abstraction.
//
// Provides a platform-agnostic interface for reading per-thread CPU cycle
// counts directly from hardware performance monitoring units (PMU). When
// available, these counts are P-state immune: a hardware cycle is a cycle
// regardless of the clock frequency at which it ticked. This eliminates
// the main source of error in CPI measurements on systems where the CPU
// frequency is not pinned (all modern macOS and Windows ARM64 devices).
//
// Platform support:
//   macOS (Apple Silicon): kpc framework in libsystem_kernel.dylib.
//     Fixed counter 0 = CPU cycles (confirmed on M1/M2/M3 by community
//     research; this is what Instruments, `time -l`, and Xcode use).
//     kpc_force_all_ctrs_set() may require root; the implementation
//     attempts it and continues gracefully on failure.
//   Windows ARM64: not yet implemented (no reliable in-process per-thread
//     cycle counter; QueryProcessorCycleTime uses 100ns units, not cycles).
//   Linux: not yet implemented (perf_event_open is the right path).
//
// Typical usage:
//
//   // At startup (once):
//   if (cycle_counter_init())
//       printf("PMU cycle counting enabled\n");
//
//   // Around a workload:
//   const uint64_t c0 = cycle_counter_read();
//   do_work();
//   const uint64_t c1 = cycle_counter_read();
//   const uint64_t cycles = c1 - c0;   // exact, P-state immune

#include <cstdint>

namespace arm64bench {

// Attempt to initialise PMU cycle counting for this process/thread.
// Returns true if hardware cycle counting is now available.
// Safe to call multiple times; subsequent calls are no-ops and return the
// same value as the first call.
bool cycle_counter_init();

// Returns true iff cycle_counter_read() will return meaningful per-thread
// CPU cycle counts.
bool cycle_counter_available();

// Read the current accumulated CPU cycle count for the calling thread.
// Returns 0 if !cycle_counter_available().
// Two consecutive calls surrounding a workload give a hardware-accurate
// cycle delta for that workload.
uint64_t cycle_counter_read();

} // namespace arm64bench
