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
// Two sources are combined:
//
//   1. What the OS reports.
//      macOS    sysctlbyname("hw.optional.arm.FEAT_XXX") — complete and reliable.
//      Linux    getauxval(AT_HWCAP / AT_HWCAP2) bits.
//      Windows  the ID_AA64ISAR0/ISAR1/PFR0_EL1 values the kernel mirrors into
//               the registry (HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0,
//               values "CP 4030" / "CP 4031" / "CP 4020"), decoded field by
//               field; IsProcessorFeaturePresent(PF_ARM_*) where that is the
//               authority (SVE/SVE2, which need OS state save support). A
//               feature with neither source is Unknown.
//
//   2. An instruction probe. For every feature that is a plain instruction
//      set extension (everything except SVE/SVE2/SME, whose usability is the
//      OS's call), one representative instruction is JIT'd and executed once
//      under an illegal-instruction trap (SEH on Windows, a SIGILL handler
//      with sigsetjmp elsewhere). A feature the OS reports present but whose
//      instruction faults is treated as absent, with a warning — that is the
//      case that otherwise kills the process mid-test with SIGILL /
//      0xC000001D. An Unknown feature is decided by the probe alone.
//
// The probe runs for every feature on the first query (one handler install,
// one pass) and results are cached; call cpu_has() from the main thread
// before starting any other threads.

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

// True iff the feature can be used by JIT'd code in this process: the OS
// reports it (or has no opinion) and its probe instruction executes.
bool cpu_has(CpuFeature f);

// The OS's answer alone, before the instruction probe.
enum class FeatureReport : uint8_t { Absent, Present, Unknown };
FeatureReport cpu_os_reports(CpuFeature f);

// Result of executing the feature's probe instruction:
//   1  executed normally
//   0  raised an illegal-instruction trap
//  -1  no probe is defined for this feature (SVE/SVE2/SME)
int cpu_probe(CpuFeature f);

// Architectural name, e.g. "FEAT_DotProd". For skip messages.
const char* cpu_feature_name(CpuFeature f);

} // namespace arm64bench
