#pragma once
// cpu_features.h
// Runtime CPU feature detection for gating JIT-emitted test sections.
//
// All test code is JIT-emitted, so the host compiler never needs to know
// whether the CPU has a feature — only the running process does. Never use
// __ARM_FEATURE_* compile-time macros to gate a test section: on a toolchain
// whose default -march lacks the feature the section silently disappears
// from the build, and CI cannot tell the difference.
//
// Sources, per platform:
//   macOS    sysctlbyname("hw.optional.arm.FEAT_XXX") — complete and reliable.
//   Linux    getauxval(AT_HWCAP / AT_HWCAP2) bits.
//   Windows  IsProcessorFeaturePresent(PF_ARM_*) where a flag exists; the
//            coverage is patchy, so features with no flag fall back to a
//            per-feature default (true for anything every Windows-on-Arm
//            core has shipped with, false otherwise). See cpu_features.cpp.
//
// Results are cached after the first query.

#include <cstdint>

namespace arm64bench {

enum class CpuFeature : uint8_t {
    AES,        // FEAT_AES    — AESE/AESD/AESMC/AESIMC
    PMULL,      // FEAT_PMULL  — PMULL/PMULL2 on poly64
    SHA256,     // FEAT_SHA256 — SHA256H/SHA256SU0/...
    SHA3,       // FEAT_SHA3   — EOR3/RAX1/XAR/BCAX
    SHA512,     // FEAT_SHA512 — SHA512H/SHA512H2/SHA512SU0/SHA512SU1
    CRC32,      // FEAT_CRC32
    LSE,        // FEAT_LSE    — LDADD/SWP/CAS...
    DotProd,    // FEAT_DotProd — SDOT/UDOT
    FP16,       // FEAT_FP16   — half-precision vector arithmetic (FMLA v8h)
    FHM,        // FEAT_FHM    — FMLAL/FMLSL (f16→f32 widening)
    JSCVT,      // FEAT_JSCVT  — FJCVTZS
    I8MM,       // FEAT_I8MM   — USDOT/SMMLA/UMMLA/USMMLA
    BF16,       // FEAT_BF16   — BFDOT/BFMMLA/BFMLAL
    LRCPC,      // FEAT_LRCPC  — LDAPR
    LRCPC2,     // FEAT_LRCPC2 — LDAPUR/STLUR
    LRCPC3,     // FEAT_LRCPC3 — LDIAPP/STILP
    SVE,        // FEAT_SVE    — scalable vectors (non-streaming)
    SVE2,       // FEAT_SVE2
    SME,        // FEAT_SME    — streaming SVE mode + ZA (Apple M4+, Cortex-X/A 2023+)

    Count_
};

// True iff the running CPU (as reported by the OS) has the feature.
bool cpu_has(CpuFeature f);

// Architectural name, e.g. "FEAT_DotProd". For skip messages.
const char* cpu_feature_name(CpuFeature f);

} // namespace arm64bench
