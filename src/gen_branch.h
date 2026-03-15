#pragma once
// gen_branch.h
// Branch prediction microbenchmark generator for arm64bench.
//
// Tests in this module probe the front-end branch prediction hardware:
//
//   RSB (Return Stack Buffer) depth sweep — call chains of increasing depth.
//     Each depth-N test executes N nested BL/RET pairs per loop iteration.
//     When N exceeds the RSB capacity, the outermost returns fall back to the
//     branch target buffer or indirect predictor, paying a misprediction
//     penalty. The depth at which per-pair latency rises identifies RSB size.
//
//   Indirect branch predictor capacity sweep — a single BLR instruction that
//     cycles through N distinct target addresses. The predictor must track
//     N different (branch_site → target) associations. When N exceeds the
//     predictor's capacity, some associations are evicted and the misprediction
//     rate rises. The knee in the latency curve identifies predictor capacity.
//
//   Conditional branch throughput — CBNZ and TBZ/TBNZ variants that are
//     always-not-taken, always-taken, or alternating. Reveals the throughput
//     ceiling for predictable conditional branches and the cost of a
//     systematically alternating (TNTN...) pattern.

#include "harness.h"

namespace arm64bench::gen {

void run_branch_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
