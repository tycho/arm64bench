#pragma once
// affinity.h
// Thread placement for the multi-thread tests.
//
// Linux and Windows can pin a thread to one logical CPU; macOS has no
// affinity API for user threads (thread_affinity_policy is a hint that
// Apple Silicon ignores), so the only placement control there is QoS class,
// which steers a thread onto the performance or efficiency cluster.

#include <cstdint>

namespace arm64bench {

// True when pin_thread_to_cpu() can actually pin (Linux, Windows).
bool affinity_supported();

// Logical CPUs this process may run on, in ascending id order. Writes at
// most `max` ids to `out` and returns the number available (which may
// exceed `max`). On macOS returns the online CPU count with ids 0..n-1,
// for sizing only — they cannot be pinned.
uint32_t allowed_cpus(uint32_t* out, uint32_t max);

// Restrict the calling thread to one logical CPU. Returns false where
// unsupported or refused.
bool pin_thread_to_cpu(uint32_t cpu);

// Undo pin_thread_to_cpu() for the calling thread.
void unpin_thread();

// Ask the scheduler to keep the calling thread on the performance cluster
// (`efficiency == false`) or the efficiency cluster (`efficiency == true`).
// macOS QoS classes; a no-op elsewhere.
void set_thread_cluster_hint(bool efficiency);

} // namespace arm64bench
