#pragma once
// gen_sve.h
// SVE / SVE2 microbenchmarks.
//
// Runs in one of two modes, chosen at runtime:
//
//   native     FEAT_SVE present (Neoverse N2, Cortex-X/A, Oryon): plain SVE
//              instructions at the implementation's vector length.
//   streaming  no FEAT_SVE but FEAT_SME present (Apple M4/M5): every test
//              function enters streaming SVE mode with SMSTART SM, runs the
//              SVE instructions at the streaming vector length (512 bits on
//              M4/M5) on the SME unit, and leaves with SMSTOP SM. This is
//              the only way to execute SVE code on Apple Silicon, and it
//              measures the SME unit rather than the core's NEON pipes.
//
// Skipped entirely, with a note, when neither is present.
//
// Tests: vector length; ADD/FADD/FMUL/FMLA/SDOT latency and chain-sweep
// throughput; FADDV reduction latency; contiguous LD1W/ST1W and LDR/STR z
// L1 throughput; WHILELT/PTRUE predicate throughput.

#include "harness.h"

namespace arm64bench::gen {

void run_sve_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
