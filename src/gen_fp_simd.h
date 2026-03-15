#pragma once
// gen_fp_simd.h
// Floating-point and NEON SIMD microbenchmark generator for arm64bench.
//
// Tests in this module characterize the FP and vector execution pipelines:
//
//   Scalar FP (f32, f64):
//     FADD, FMUL, FMADD latency and throughput sweeps. Reveals the number
//     of FP execution units and whether FMADD has a dedicated accumulator
//     forwarding path (analogous to the integer MADD result we saw).
//     FDIV and FSQRT latency — these are on a separate, non-pipelined unit.
//
//   NEON vector (4×f32, 2×f64):
//     Same tests as scalar but on 128-bit vector registers. If the scalar
//     and vector FP units share hardware, the throughput floors will match.
//     If vector FMUL/FADD saturate at a lower chain count than scalar, the
//     architecture has separate scalar and vector pipelines.
//
//   Integer NEON (4×i32, 8×i16):
//     Integer vector ADD and MUL throughput. Reveals whether integer NEON
//     shares execution units with FP NEON or has dedicated vector ALU ports.
//
//   Mixed port pressure:
//     FADD + FMUL interleaved — tests whether multiply and add share ports.
//     NEON fadd + scalar fadd — tests scalar vs vector dispatch separation.

#include "harness.h"

namespace arm64bench::gen {

void run_fp_simd_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
