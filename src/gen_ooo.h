#pragma once
// gen_ooo.h
// Out-of-order window sizing: reorder buffer, physical register files,
// load queue, and store queue.
//
// Method (Henry Wong's two-miss probe): two independent pointer chains, each
// a cache miss to DRAM, interleaved with N filler instructions:
//
//     ldr x0, [x0]      ← miss A
//     <N fillers>
//     ldr x1, [x1]      ← miss B (independent of A)
//     <N fillers>
//
// While N + 2 fits in the out-of-order window, miss B issues in the shadow
// of miss A and the pair costs about one DRAM latency. Once the fillers
// exhaust the window before B can enter it, B waits for A to retire and the
// pair costs two. The N at which the time per pair jumps is the capacity of
// whichever structure the filler consumes:
//
//   NOP                     reorder-buffer entries only
//   ADD  xN, xN, #1         ROB + an integer physical register each
//   FADD dN, dN, dC         ROB + an FP/SIMD physical register each
//   LDR  xzr, [x9] (L1 hit) ROB + a load-queue entry each (no register written)
//   STR  x2, [x9, #k]       ROB + a store-queue entry each
//   CMP  x2, x3             ROB + a flag (NZCV) physical register each
//   B.NE (not taken)        ROB + a branch-order-buffer entry each
//
// The smallest of ROB and the filler's own structure wins, so the ADD/FADD
// knees only reveal the register files where they are smaller than the ROB
// (true on Apple M-series), and the load/store knees are the queue sizes.

#include "harness.h"

namespace arm64bench::gen {

void run_ooo_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
