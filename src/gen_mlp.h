#pragma once
// gen_mlp.h
// Memory-level parallelism: how many cache misses can be in flight at once
// at each level of the hierarchy.
//
// K independent random pointer chases are interleaved over one buffer whose
// size selects the level that services the misses (L2, far L2/SLC, DRAM).
// Each loop iteration advances every chain by one load, so the K loads are
// independent and the out-of-order core can overlap them. While the level's
// miss-handling structures have room, the iteration costs about one miss
// latency regardless of K and the time per load falls as 1/K; once they are
// full the iteration time grows with K and the time per load flattens. The
// K at the flattening is the number of overlapped misses that level sustains.
//
// Reported per test: ns per load and the equivalent cache-line bandwidth.

#include "harness.h"

namespace arm64bench::gen {

void run_mlp_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
