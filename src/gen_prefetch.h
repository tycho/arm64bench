#pragma once
// gen_prefetch.h
// Hardware prefetcher characterization and software-prefetch (PRFM)
// effectiveness.
//
//   stride streams    dependent loads in runs of 32 accesses at a fixed
//                     stride, forward or backward, over a DRAM-sized buffer.
//                     The runs tile the buffer so every line is visited once
//                     per pass whatever the stride is: the footprint is
//                     constant and only the pattern changes. Per-load time
//                     well under the random-chase latency means the hardware
//                     prefetcher is following the stream; the strides at which
//                     it climbs back are the ones the prefetcher cannot
//                     follow (too large, a page boundary it will not cross,
//                     or a stride that does not tile whatever region a
//                     spatial prefetcher learns — the sweep includes
//                     non-power-of-two strides to tell those apart).
//
//   PRFM lookahead    a random DRAM pointer chase whose nodes also hold the
//                     address D hops ahead, prefetched with PRFM PLDL1KEEP
//                     before the dependent load. Per-load time versus D shows
//                     whether PRFM is honored at all and how far ahead it
//                     needs to be issued to hide the miss.

#include "harness.h"

namespace arm64bench::gen {

void run_prefetch_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
