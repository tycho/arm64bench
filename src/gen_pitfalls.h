#pragma once
// gen_pitfalls.h
// Microarchitectural pitfall tests for arm64bench.
//
// Tests in this module probe specific performance hazards that differ
// significantly between Apple M-series and Snapdragon/Cortex cores:
//
//   Store-to-load forwarding — width-matched, width-mismatched, and
//     partially-overlapping store+load sequences. Apple Silicon and
//     Qualcomm handle mismatches differently; a mismatch that costs
//     ~1 cycle on Oryon may cost ~10 cycles on M1.
//
//   Memory ordering — LDAR vs LDR latency, DMB/DSB/ISB cost. These are
//     invisible in the common case but dominate lock-heavy code paths.
//
//   Non-temporal stores (STNP) — at DRAM sizes, STNP bypasses the cache
//     and eliminates write-allocate read-for-ownership traffic. This can
//     nearly double write bandwidth for streaming kernels.
//
//   Misaligned loads — penalty for loads that cross cache-line or page
//     boundaries. Varies dramatically between cores.
//
//   CAS latency — the true cost of a single compare-and-swap on each
//     platform, which bounds the throughput of single-threaded spinlocks.
//
//   LRCPC load-acquire (FEAT_LRCPC / FEAT_LRCPC2) — LDAPR and LDAPUR latency
//     chains vs LDR and LDAR. Exposes incorrect LRCPC implementations where
//     the CPU treats LDAPR as a full LDAR. Also tests store-to-load forwarding
//     through STLR/STLUR → LDAPR/LDAPUR pairs.

#include "harness.h"

namespace arm64bench::gen {

void run_pitfall_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
