#pragma once
// gen_crypto.h
// ARMv8-A cryptography extension tests: AESE/AESMC (and their fusion),
// PMULL poly64, SHA256H/SHA256SU0, CRC32B/W/X. Uses base_params.loops and
// instructions_per_loop as the loop shape.

#include "harness.h"

namespace arm64bench::gen {

void run_crypto_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
