#pragma once
// gen_i8mm.h
// FEAT_I8MM tests: USDOT latency/throughput and SMMLA/UMMLA/USMMLA
// latency/throughput. Prints a skip line and returns when the CPU lacks
// FEAT_I8MM (checked at runtime via cpu_has).

#include "harness.h"

namespace arm64bench::gen {

void run_i8mm_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
