#pragma once
// gen_integer.h
// Integer ALU microbenchmark generator for arm64bench.
//
// Tests in this module probe:
//   - Instruction latency (chained dependency, 1 result register)
//   - Instruction throughput (N independent chains, sweeping N to
//     saturate execution units and reveal their count)
//   - Mixed instruction streams (heterogeneous port pressure)
//
// All test functions are JIT-generated via AsmJit and compiled through
// g_jit_pool. Each function is released immediately after benchmarking.

#include "harness.h"

namespace arm64bench::gen {

// Run the full integer ALU test suite using base_params for sampling
// configuration. Loop counts and unroll factors for specific instruction
// classes (e.g. divide) are derived from base_params internally.
void run_integer_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
