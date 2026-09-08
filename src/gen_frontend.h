#pragma once
// gen_frontend.h
// Front-end and rename-stage probes: things that happen to an instruction
// before it reaches an execution unit.
//
//   NOP throughput          decode/rename width, and fetch bandwidth once the
//                           loop body outgrows the fetch window
//   MOV elimination         is `mov xN, xM` (and FMOV/ORR for FP/vector)
//                           resolved at rename with zero latency?
//   zero idioms             does `eor x0, x0, x0` / `sub x0, x0, x0` /
//                           `eor v0, v0, v0` break the dependency on x0/v0?
//   macro-op fusion         CMP+B.cond, ADRP+ADD, MOVZ+MOVK: does a pair
//                           issue as one micro-op (pairs/clk vs singles/clk)?
//   branch throughput       taken unconditional and not-taken conditional
//   ISB                     pipeline flush cost
//
// Every test here is baseline ARMv8-A; nothing is feature-gated.

#include "harness.h"

namespace arm64bench::gen {

void run_frontend_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
