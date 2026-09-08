#pragma once
// gen_icache.h
// Instruction-side memory hierarchy: I-cache capacity and fetch bandwidth,
// and instruction-TLB reach.
//
//   code-size sweep   straight-line NOP bodies from 16 KB to 8 MB. NOPs
//                     decode at the front-end's full width when fetched from
//                     L1I, so clk/insn rises exactly where the body outgrows
//                     each level: L1I, then L2, then DRAM-fed fetch.
//
//   page-chain sweep  N taken branches, each on its own 16 KB page, each
//                     jumping to the next. One I-cache line per page, so the
//                     I-cache never fills; what grows with N is the number
//                     of instruction pages in flight. clk per branch rises
//                     where N outgrows the L1 iTLB, then the L2 TLB.
//
// JIT-emitted code is what makes this cheap: the bodies are generated, not
// compiled, so a 16 MB function or 2048 pages of one-branch stubs cost
// nothing but memory.

#include "harness.h"

namespace arm64bench::gen {

void run_icache_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
