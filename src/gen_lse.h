#pragma once
// gen_lse.h
// LSE (Large System Extensions, ARMv8.1) atomic microbenchmark generator.
//
// LSE provides single-instruction atomic read-modify-write operations,
// eliminating the LL/SC retry loop required by the original ARMv8.0 exclusive
// pair (LDXR / STXR). This module characterises the full family of LSE RMW
// instructions and compares them against the LL/SC baseline.
//
// ── What is measured ─────────────────────────────────────────────────────────
//
// All tests are single-threaded and operate on a single 64-byte-aligned word
// that lives permanently in L1 D-cache (it is touched by the warm-up calls
// before timing begins). The measured clk/op is therefore the round-trip
// latency for a serialised atomic RMW on a resident cache line — this
// captures:
//   1. The micro-architectural cost of the load → RMW → store pipeline for
//      that instruction class.
//   2. The barrier cost imposed by acquire (A) or release (L) semantics.
//
// The tests do NOT measure coherence-protocol traffic (that would require a
// second thread on a different core holding the cache line).
//
// ── Architecture differentiation ─────────────────────────────────────────────
//
// Apple M-series (Firestorm / Everest):
//   LDADDAL ≈ 4–6 cycles — extremely low ordering barrier overhead.
//   CASAL   ≈ 4–6 cycles — same fast path.
//   LDAXR+STLXR ≈ 4–6 cycles — similar to LSE; hardware fuses the pair.
//
// Qualcomm Oryon (Snapdragon X Elite):
//   Values not yet published; expected to differ from Apple.
//   A key question: is LDADDAL faster or slower than CASAL?
//   Does STADDL (no return) have a different cost from LDADDL?
//
// Older ARM cores (Cortex-A76/A78/X1):
//   LDADDAL typically 10–20 cycles.
//   LDAXR+STLXR similar or slightly slower.
//
// ── Instructions covered ─────────────────────────────────────────────────────
//
//   LDADD / LDADDA / LDADDL / LDADDAL   — load-add, 4 orderings
//   STADD / STADDL                       — store-add (no return), 2 orderings
//   SWP / SWPA / SWPL / SWPAL            — exchange, 4 orderings
//   CAS / CASAL                          — compare-and-swap, 2 orderings
//   LDAXR + STLXR                        — LL/SC baseline (full ordering)

#include "harness.h"

namespace arm64bench::gen {

void run_lse_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
