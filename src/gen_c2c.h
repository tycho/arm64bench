#pragma once
// gen_c2c.h
// Core-to-core communication: cache-line transfer latency between two
// threads on different cores, and the cost of a contended atomic.
//
// A partner thread runs a JIT'd responder while the main thread runs the
// timed function; the pair is placed on a chosen core pair where the OS
// can pin threads (Linux, Windows), or steered by QoS class on macOS
// (performance cluster vs efficiency cluster), which has no affinity API.
//
//   1-line round trip   one shared line holds a counter. The main thread
//                       writes an odd value and spins until the partner has
//                       written it back +1; the partner spins until it sees
//                       an odd value and replies. One round trip is two
//                       line migrations. Measured with LDAR/STLR and with
//                       plain LDR/STR.
//
//   2-line round trip   the main thread writes its own line and spins on
//                       the partner's; the partner mirrors. Each line has one
//                       writer, so the two migrations do not serialize on a
//                       single line's write-after-read ownership change.
//
//   contended LDADDAL   both threads increment the same word as fast as they
//                       can; the main thread's ns per increment against its
//                       uncontended cost is the price of the line bouncing.
//
// The reported ns is the round trip (two one-way transfers) or one
// increment; clk~ is in the main thread's core clocks.

#include "harness.h"

namespace arm64bench::gen {

void run_c2c_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
