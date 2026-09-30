#pragma once
// gen_fpenv.h
// Floating-point environment tests: what it costs to read and write FPCR and
// FPSR, and what a write does to the FP code around it.
//
// The motivating user is a binary translator that runs x86 SSE/AVX floating
// point natively on NEON. It has to put the guest's MXCSR into the host FPCR
// (rounding mode → RMode, FTZ → FZ, and with FEAT_AFP: DAZ → FIZ, AH for
// x86 NaN/min/max rules, NEP for x86 scalar merging) while generated code
// runs, put the host value back around helpers that run C floating point,
// and collect exception flags from FPSR. Whether that is a per-instruction,
// per-block or per-helper cost depends on numbers no optimisation guide
// gives: is MSR FPCR renamed or does it drain the pipeline, does a write of
// an unchanged value cost the same as a real change, does MRS FPSR wait for
// every FP op ahead of it.
//
// Every JIT'd function here saves FPCR in its setup and restores it in its
// teardown, so the harness, the reference function and the C++ caller never
// run with a modified FPCR. Trap-enable bits are never set.
//
// AFP rows (FIZ, AH, NEP: FPCR bits 0, 1, 2) are skipped with a note on a
// core without FEAT_AFP.

#include "harness.h"

namespace arm64bench::gen {

void run_fpenv_tests(const BenchmarkParams& base_params);

} // namespace arm64bench::gen
