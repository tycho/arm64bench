#pragma once
// gen_bf16.h
// FEAT_BF16 tests: BFDOT (bf16 dot product into f32), BFMMLA (2×4 × 4×2
// bf16 matrix multiply-accumulate into f32), and BFMLALB/BFMLALT (widening
// multiply-accumulate of the bottom/top bf16 halves).
//
// The question mirrors the I8MM one: is BFMMLA real matrix hardware (more
// MACs per cycle than BFDOT) or two BFDOT micro-ops (same MAC rate, twice
// the latency)? Skips itself with a note when the CPU lacks FEAT_BF16.

#include "harness.h"

namespace arm64bench::gen {

void run_bf16_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
