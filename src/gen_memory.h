#pragma once
// gen_memory.h
// Memory hierarchy microbenchmark generator for arm64bench.
//
// Tests in this module characterize the cache and memory subsystem:
//
//   Load latency sweep  — random pointer-chase through buffers of increasing
//     size. Each load serializes on the previous result, so the prefetcher
//     cannot help. The latency at each buffer size reveals the access time
//     for whichever cache level (or DRAM) the buffer fits into. The size at
//     which latency jumps identifies the cache capacity boundaries.
//
//   Sequential load bandwidth sweep — stride-64 loads through the buffer with
//     8 independent LDP streams per step. The hardware prefetcher engages for
//     sequential patterns, so this measures *peak sustained read bandwidth*
//     at each cache level, not latency.
//
//   Sequential store bandwidth sweep — symmetric with load bandwidth, using
//     STP with 8 streams per step. Reveals write bandwidth, which is often
//     asymmetric with read bandwidth (e.g. due to write-allocate policy).

#include "harness.h"

namespace arm64bench::gen {

void run_memory_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
