#pragma once
// affinity.h
// Thread placement and CPU topology.
//
// Linux and Windows can pin a thread to a set of logical CPUs; macOS has no
// affinity API for user threads (thread_affinity_policy is a hint that
// Apple Silicon ignores), so the only placement control there is QoS class,
// which steers a thread onto the performance or efficiency cluster.
//
// The topology query exists because every heterogeneous ARM machine puts a
// different core on cpu 0: the Snapdragon X2 Elite enumerates its six 3.6 GHz
// Performance cores first and its twelve 5 GHz Prime cores after them, the
// 8cx Gen 3 puts the Cortex-X1C cores first. Left to float, a run mixes core
// types test by test. cpu_select.h uses this to pin the main thread to one
// cluster.

#include <cstdint>

namespace arm64bench {

constexpr uint32_t kMaxCpus = 256;

struct CpuInfo {
    uint32_t id;
    int32_t  cluster;     // index of the L2-sharing group; -1 if unknown
    int32_t  perf_class;  // OS efficiency class, higher = faster; -1 if unknown
    uint32_t max_khz;     // OS-reported maximum clock; 0 if unknown
    uint64_t midr;        // MIDR_EL1; 0 if unknown
};

struct CpuTopology {
    uint32_t count;
    bool     have_clusters;   // at least one CPU has a cluster id
    bool     have_classes;    // at least one CPU has a perf class
    CpuInfo  cpus[kMaxCpus];
};

// True when threads can actually be pinned (Linux, Windows).
bool affinity_supported();

// Logical CPUs this process may run on, in ascending id order: the calling
// thread's set the first time this is called (before main() pins anything)
// and cached from then on. Writes at most `max` ids to `out` and returns the
// number available (which may exceed `max`). On macOS returns the online
// CPU count with ids 0..n-1, for sizing only — they cannot be pinned.
uint32_t allowed_cpus(uint32_t* out, uint32_t max);

// The calling thread's current affinity set, same conventions.
uint32_t thread_cpus(uint32_t* out, uint32_t max);

// Describe every CPU in allowed_cpus(). Fields the OS cannot supply are left
// at their "unknown" values; returns false only where nothing is known.
bool cpu_topology(CpuTopology& out);

// Restrict the calling thread to one logical CPU, or to a set. Returns false
// where unsupported or refused.
bool pin_thread_to_cpu(uint32_t cpu);
bool pin_thread_to_cpus(const uint32_t* cpus, uint32_t n);

// Undo any pin for the calling thread (back to the process's full set).
void unpin_thread();

// Ask the scheduler to keep the calling thread on the performance cluster
// (`efficiency == false`) or the efficiency cluster (`efficiency == true`).
// macOS QoS classes; a no-op elsewhere.
void set_thread_cluster_hint(bool efficiency);

} // namespace arm64bench
