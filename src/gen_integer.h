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

// Bitfield insert / extract instruction tests (BFI, BFXIL, UBFX, SBFX).
// Includes Darek Mihocka's overlapping-bitfield BFI chain (rotated across
// three registers with bit positions that overlap) — a stress designed to
// defeat any micro-architectural attempt to break the destination dependency
// that BFI inherently has (it must read Xd to preserve non-inserted bits).
//
// Latency interpretation:
//   1 clk/insn ≈ Apple Silicon, Cortex-X1+, Snapdragon X/X2 (Oryon).
//   2 clk/insn ≈ pre-X1 Cortex-A and pre-Oryon Snapdragon — historical.
void run_bitfield_tests(const BenchmarkParams& base_params);

// Create a JIT function that runs a chained ADD latency loop suitable for
// use as the Tier 2 ratio normalization reference. Each iteration executes
// (unroll) ADD instructions in a serial dependency chain (each output feeds
// the next input), so the expected throughput is 1 cycle per instruction on
// all modern ARM64 cores. The caller owns the returned pointer; release it
// via g_jit_pool->release() when done (typically at process exit).
TestFn create_add_latency_ref(uint64_t loops, uint32_t unroll);

} // namespace arm64bench::gen
