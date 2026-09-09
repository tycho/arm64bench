#pragma once
// cpu_select.h
// Choose the CPUs the main thread runs on, and pin it there.
//
// On a heterogeneous machine a floating thread lands on a different core
// type test by test (the X2 Elite results mixed 5 GHz Prime and 3.6 GHz
// Performance cores; the 8cx Gen 3 mixed Cortex-X1C and A78C), and the
// numbering does not help: cpu 0 is a small core on the X2 Elite. So the
// default is measured, not assumed: every allowed CPU runs a pinned
// dependent-ADD chain for a few milliseconds, which reads its clock
// directly (ADD is one cycle everywhere), and the main thread is pinned to
// the L2-sharing cluster with the highest clock. Floating within the
// cluster is allowed; it costs at most an L1 refill, which the per-sample
// warm-up covers.
//
//   --cpu auto   (default) fastest cluster by measured clock
//   --cpu N      the cluster containing logical CPU N
//   --cpu any    no pinning (the pre-2026-09 behaviour)
//
// macOS has no affinity API; the QoS class already keeps the thread on the
// performance cluster and this module only prints that.

#include "affinity.h"

#include <cstdint>

namespace arm64bench {

constexpr int kCpuAuto = -2;
constexpr int kCpuAny  = -1;

struct CpuChoice {
    uint32_t cpus[kMaxCpus];
    uint32_t n;        // 0 when nothing was pinned
    bool     pinned;
};

// Parse "auto" / "any" / "<n>" into kCpuAuto / kCpuAny / n. Returns false
// for anything else.
bool parse_cpu_arg(const char* s, int* mode);

// Query the topology, survey the clocks (unless `survey` is false, e.g. in
// smoke mode), print a summary, and pin the calling thread. Needs the JIT
// pool; call after g_jit_pool is set and before calibration.
void select_cpus(int mode, bool survey, CpuChoice& out);

} // namespace arm64bench
