// cpu_features.cpp
// Runtime CPU feature detection. See cpu_features.h for the rationale.
//
// Each feature carries the OS-specific handle for every platform:
//   macOS    the hw.optional.arm.FEAT_* sysctl name (missing name → false).
//   Linux    an AT_HWCAP or AT_HWCAP2 bit. Values are the kernel's; we define
//            them locally so an old <asm/hwcap.h> cannot hide a newer bit.
//   Windows  a PF_ARM_* IsProcessorFeaturePresent() flag, or -1 when the SDK
//            has none, in which case `win_default` is used. Windows-on-Arm
//            has only ever shipped on cores at ARMv8.2+ with the full
//            ARMv8.4 optional set (Snapdragon 8cx/X, Ampere, Cobalt 100), so
//            the default is true for those and false for anything newer.

#include "cpu_features.h"

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sys/auxv.h>
#endif

namespace arm64bench {

namespace {

// ── Linux HWCAP bits (arch/arm64/include/uapi/asm/hwcap.h) ──────────────────
constexpr uint64_t kHwcapAES      = 1ULL << 3;
constexpr uint64_t kHwcapPMULL    = 1ULL << 4;
constexpr uint64_t kHwcapSHA2     = 1ULL << 6;
constexpr uint64_t kHwcapCRC32    = 1ULL << 7;
constexpr uint64_t kHwcapATOMICS  = 1ULL << 8;
constexpr uint64_t kHwcapASIMDHP  = 1ULL << 10;
constexpr uint64_t kHwcapJSCVT    = 1ULL << 13;
constexpr uint64_t kHwcapLRCPC    = 1ULL << 15;
constexpr uint64_t kHwcapILRCPC   = 1ULL << 16;
constexpr uint64_t kHwcapSHA3     = 1ULL << 17;
constexpr uint64_t kHwcapASIMDDP  = 1ULL << 20;
constexpr uint64_t kHwcapASIMDFHM = 1ULL << 23;
constexpr uint64_t kHwcap2I8MM    = 1ULL << 13;
constexpr uint64_t kHwcap2BF16    = 1ULL << 14;
constexpr uint64_t kHwcap2LRCPC3  = 1ULL << 46;
constexpr uint64_t kHwcapSVE      = 1ULL << 22;
constexpr uint64_t kHwcap2SVE2    = 1ULL << 1;
constexpr uint64_t kHwcap2SME     = 1ULL << 23;

// ── Windows PF_ARM_* flags (winnt.h) ────────────────────────────────────────
// Defined locally so older SDKs still build; the values are ABI.
constexpr int kPfCrypto = 30;   // PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE
constexpr int kPfCrc32  = 31;   // PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE
constexpr int kPfAtomic = 34;   // PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE
constexpr int kPfDotProd= 43;   // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
constexpr int kPfJscvt  = 44;   // PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE
constexpr int kPfLrcpc  = 45;   // PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE
constexpr int kPfSve    = 46;   // PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
constexpr int kPfSve2   = 47;   // PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE
// No direct I8MM flag exists; SVE-I8MM implies I8MM (used by FFmpeg/dav1d).
#if defined(PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE)
constexpr int kPfI8mm   = PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE;
#else
constexpr int kPfI8mm   = -1;
#endif
constexpr int kPfNone   = -1;

struct FeatureInfo {
    const char* name;          // architectural name
    const char* sysctl;        // macOS: hw.optional.arm.<sysctl>
    uint64_t    hwcap;         // Linux AT_HWCAP bit (0 = none)
    uint64_t    hwcap2;        // Linux AT_HWCAP2 bit (0 = none)
    int         win_pf;        // Windows PF_* flag, or -1
    bool        win_default;   // Windows: used when win_pf == -1
};

constexpr FeatureInfo kFeatures[] = {
    // enum order must match CpuFeature.
    { "FEAT_AES",     "FEAT_AES",     kHwcapAES,      0,             kPfCrypto,  true  },
    { "FEAT_PMULL",   "FEAT_PMULL",   kHwcapPMULL,    0,             kPfCrypto,  true  },
    { "FEAT_SHA256",  "FEAT_SHA256",  kHwcapSHA2,     0,             kPfCrypto,  true  },
    { "FEAT_SHA3",    "FEAT_SHA3",    kHwcapSHA3,     0,             kPfNone,    true  },
    { "FEAT_CRC32",   "FEAT_CRC32",   kHwcapCRC32,    0,             kPfCrc32,   true  },
    { "FEAT_LSE",     "FEAT_LSE",     kHwcapATOMICS,  0,             kPfAtomic,  true  },
    { "FEAT_DotProd", "FEAT_DotProd", kHwcapASIMDDP,  0,             kPfDotProd, true  },
    { "FEAT_FP16",    "FEAT_FP16",    kHwcapASIMDHP,  0,             kPfNone,    true  },
    { "FEAT_FHM",     "FEAT_FHM",     kHwcapASIMDFHM, 0,             kPfNone,    true  },
    { "FEAT_JSCVT",   "FEAT_JSCVT",   kHwcapJSCVT,    0,             kPfJscvt,   true  },
    { "FEAT_I8MM",    "FEAT_I8MM",    0,              kHwcap2I8MM,   kPfI8mm,    true  },
    { "FEAT_BF16",    "FEAT_BF16",    0,              kHwcap2BF16,   kPfNone,    true  },
    { "FEAT_LRCPC",   "FEAT_LRCPC",   kHwcapLRCPC,    0,             kPfLrcpc,   true  },
    { "FEAT_LRCPC2",  "FEAT_LRCPC2",  kHwcapILRCPC,   0,             kPfNone,    true  },
    { "FEAT_LRCPC3",  "FEAT_LRCPC3",  0,              kHwcap2LRCPC3, kPfNone,    false },
    { "FEAT_SVE",     "FEAT_SVE",     kHwcapSVE,      0,             kPfSve,     false },
    { "FEAT_SVE2",    "FEAT_SVE2",    0,              kHwcap2SVE2,   kPfSve2,    false },
    { "FEAT_SME",     "FEAT_SME",     0,              kHwcap2SME,    kPfNone,    false },
};
static_assert(sizeof(kFeatures) / sizeof(kFeatures[0]) ==
              static_cast<size_t>(CpuFeature::Count_),
              "kFeatures must have one entry per CpuFeature");

bool query(const FeatureInfo& f) {
#if defined(__APPLE__)
    char key[64] = "hw.optional.arm.";
    size_t n = 16;
    for (const char* s = f.sysctl; *s && n < sizeof(key) - 1; ++s) key[n++] = *s;
    key[n] = '\0';
    int val = 0;
    size_t len = sizeof(val);
    return sysctlbyname(key, &val, &len, nullptr, 0) == 0 && val != 0;
#elif defined(__linux__)
    if (f.hwcap)  return (getauxval(AT_HWCAP)  & f.hwcap)  != 0;
    if (f.hwcap2) return (getauxval(AT_HWCAP2) & f.hwcap2) != 0;
    return false;
#elif defined(_WIN32)
    if (f.win_pf >= 0)
        return IsProcessorFeaturePresent(static_cast<DWORD>(f.win_pf)) != 0;
    return f.win_default;
#else
    (void)f;
    return false;
#endif
}

} // namespace

bool cpu_has(CpuFeature feat) {
    // 0 = not yet queried, 1 = absent, 2 = present.
    static uint8_t cache[static_cast<size_t>(CpuFeature::Count_)] = {};
    const size_t i = static_cast<size_t>(feat);
    if (i >= static_cast<size_t>(CpuFeature::Count_)) return false;
    if (cache[i] == 0) cache[i] = query(kFeatures[i]) ? 2 : 1;
    return cache[i] == 2;
}

const char* cpu_feature_name(CpuFeature feat) {
    const size_t i = static_cast<size_t>(feat);
    return (i < static_cast<size_t>(CpuFeature::Count_)) ? kFeatures[i].name : "?";
}

} // namespace arm64bench
