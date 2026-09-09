// cpu_features.cpp
// Runtime CPU feature detection. See cpu_features.h for the rationale.
//
// Each feature carries the OS-specific handle for every platform, plus one
// probe instruction:
//   macOS    the hw.optional.arm.FEAT_* sysctl name (missing name → absent).
//   Linux    an AT_HWCAP or AT_HWCAP2 bit. Values are the kernel's; we define
//            them locally so an old <asm/hwcap.h> cannot hide a newer bit.
//   Windows  a 4-bit field of ID_AA64ISAR0_EL1 / ID_AA64ISAR1_EL1 /
//            ID_AA64PFR0_EL1, read from the registry where the kernel mirrors
//            them, or a PF_ARM_* IsProcessorFeaturePresent() flag for the
//            features whose usability depends on the OS (SVE/SVE2). Neither →
//            Unknown, decided by the probe. IsProcessorFeaturePresent alone
//            is not enough: it has no flag for SHA3/SHA512/FP16/FHM/BF16/
//            LRCPC2, and an earlier "assume present" default for those
//            crashed on a Snapdragon 8cx Gen 3 (Cortex-X1C/A78C, no SHA3)
//            with 0xC000001D.
//   probe    one instruction that only exists with the feature, executed
//            once under an illegal-instruction trap. Operates on x0 (a
//            64-byte scratch buffer), x1, x2 and v0 only — all caller-saved —
//            so a faulting probe leaves nothing to restore. 0 = no probe.
//
// Field positions follow the Arm ARM (ID_AA64ISAR0_EL1: AES[7:4] SHA2[15:12]
// CRC32[19:16] Atomic[23:20] SHA3[35:32] DP[47:44] FHM[51:48];
// ID_AA64ISAR1_EL1: JSCVT[15:12] LRCPC[23:20] BF16[47:44] I8MM[55:52];
// ID_AA64PFR0_EL1: AdvSIMD[23:20], where 0xF means no AdvSIMD at all).

#include "cpu_features.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <asmjit/core.h>
#include <asmjit/a64.h>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <csetjmp>
#  include <csignal>
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
#  include <csetjmp>
#  include <csignal>
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
constexpr uint64_t kHwcapSHA512   = 1ULL << 21;
constexpr uint64_t kHwcapASIMDDP  = 1ULL << 20;
constexpr uint64_t kHwcapASIMDFHM = 1ULL << 23;
constexpr uint64_t kHwcap2I8MM    = 1ULL << 13;
constexpr uint64_t kHwcap2BF16    = 1ULL << 14;
constexpr uint64_t kHwcap2LRCPC3  = 1ULL << 46;
constexpr uint64_t kHwcapSVE      = 1ULL << 22;
constexpr uint64_t kHwcap2SVE2    = 1ULL << 1;
constexpr uint64_t kHwcap2SME     = 1ULL << 23;

// ── Windows PF_ARM_* flags (winnt.h) ────────────────────────────────────────
// Defined locally so older SDKs still build; the values are ABI. Only the
// flags that carry information the ID registers do not (OS support for SVE
// state) are used as an authority; the rest are fallbacks for a machine
// where the registry values cannot be read.
constexpr int kPfCrypto = 30;   // PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE
constexpr int kPfCrc32  = 31;   // PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE
constexpr int kPfAtomic = 34;   // PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE
constexpr int kPfDotProd= 43;   // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
constexpr int kPfJscvt  = 44;   // PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE
constexpr int kPfLrcpc  = 45;   // PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE
constexpr int kPfSve    = 46;   // PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
constexpr int kPfSve2   = 47;   // PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE
constexpr int kPfNone   = -1;

// ── Windows registry ID register mirrors ────────────────────────────────────
enum IdReg : uint8_t { kIdNone, kIdISAR0, kIdISAR1, kIdPFR0, kIdCount_ };

struct IdField {
    uint8_t reg;    // IdReg
    uint8_t shift;  // low bit of the 4-bit field
    uint8_t min;    // present iff field >= min (and field != 0xF)
};

struct FeatureInfo {
    const char* name;          // architectural name
    const char* sysctl;        // macOS: hw.optional.arm.<sysctl>
    uint64_t    hwcap;         // Linux AT_HWCAP bit (0 = none)
    uint64_t    hwcap2;        // Linux AT_HWCAP2 bit (0 = none)
    int         win_pf;        // Windows PF_* flag, or -1
    IdField     id;            // Windows ID register field, or {kIdNone}
    uint32_t    probe;         // probe instruction encoding, 0 = none
};

constexpr FeatureInfo kFeatures[] = {
    // enum order must match CpuFeature.
    { "FEAT_AES",     "FEAT_AES",     kHwcapAES,      0,             kPfCrypto,  { kIdISAR0,  4, 1 }, 0x4E284800 }, // aese      v0.16b, v0.16b
    { "FEAT_PMULL",   "FEAT_PMULL",   kHwcapPMULL,    0,             kPfCrypto,  { kIdISAR0,  4, 2 }, 0x0EE0E000 }, // pmull     v0.1q, v0.1d, v0.1d
    { "FEAT_SHA256",  "FEAT_SHA256",  kHwcapSHA2,     0,             kPfCrypto,  { kIdISAR0, 12, 1 }, 0x5E282800 }, // sha256su0 v0.4s, v0.4s
    { "FEAT_SHA3",    "FEAT_SHA3",    kHwcapSHA3,     0,             kPfNone,    { kIdISAR0, 32, 1 }, 0xCE000000 }, // eor3      v0.16b, v0.16b, v0.16b, v0.16b
    { "FEAT_SHA512",  "FEAT_SHA512",  kHwcapSHA512,   0,             kPfNone,    { kIdISAR0, 12, 2 }, 0xCEC08000 }, // sha512su0 v0.2d, v0.2d
    { "FEAT_CRC32",   "FEAT_CRC32",   kHwcapCRC32,    0,             kPfCrc32,   { kIdISAR0, 16, 1 }, 0x9AC14C21 }, // crc32x    w1, w1, x1
    { "FEAT_LSE",     "FEAT_LSE",     kHwcapATOMICS,  0,             kPfAtomic,  { kIdISAR0, 20, 2 }, 0xF8210001 }, // ldadd     x1, x1, [x0]
    { "FEAT_DotProd", "FEAT_DotProd", kHwcapASIMDDP,  0,             kPfDotProd, { kIdISAR0, 44, 1 }, 0x4E809400 }, // sdot      v0.4s, v0.16b, v0.16b
    { "FEAT_FP16",    "FEAT_FP16",    kHwcapASIMDHP,  0,             kPfNone,    { kIdPFR0,  20, 1 }, 0x4E401400 }, // fadd      v0.8h, v0.8h, v0.8h
    { "FEAT_FHM",     "FEAT_FHM",     kHwcapASIMDFHM, 0,             kPfNone,    { kIdISAR0, 48, 1 }, 0x4E20EC00 }, // fmlal     v0.4s, v0.4h, v0.4h
    { "FEAT_JSCVT",   "FEAT_JSCVT",   kHwcapJSCVT,    0,             kPfJscvt,   { kIdISAR1, 12, 1 }, 0x1E7E0001 }, // fjcvtzs   w1, d0
    { "FEAT_I8MM",    "FEAT_I8MM",    0,              kHwcap2I8MM,   kPfNone,    { kIdISAR1, 52, 1 }, 0x4E809C00 }, // usdot     v0.4s, v0.16b, v0.16b
    { "FEAT_BF16",    "FEAT_BF16",    0,              kHwcap2BF16,   kPfNone,    { kIdISAR1, 44, 1 }, 0x6E40FC00 }, // bfdot     v0.4s, v0.8h, v0.8h
    { "FEAT_LRCPC",   "FEAT_LRCPC",   kHwcapLRCPC,    0,             kPfLrcpc,   { kIdISAR1, 20, 1 }, 0xF8BFC001 }, // ldapr     x1, [x0]
    { "FEAT_LRCPC2",  "FEAT_LRCPC2",  kHwcapILRCPC,   0,             kPfNone,    { kIdISAR1, 20, 2 }, 0xD9400001 }, // ldapur    x1, [x0]
    { "FEAT_LRCPC3",  "FEAT_LRCPC3",  0,              kHwcap2LRCPC3, kPfNone,    { kIdISAR1, 20, 3 }, 0xD9421801 }, // ldiapp    x1, x2, [x0]
    { "FEAT_SVE",     "FEAT_SVE",     kHwcapSVE,      0,             kPfSve,     { kIdNone,   0, 0 }, 0 },
    { "FEAT_SVE2",    "FEAT_SVE2",    0,              kHwcap2SVE2,   kPfSve2,    { kIdNone,   0, 0 }, 0 },
    { "FEAT_SME",     "FEAT_SME",     0,              kHwcap2SME,    kPfNone,    { kIdNone,   0, 0 }, 0 },
};
constexpr size_t kCount = static_cast<size_t>(CpuFeature::Count_);
static_assert(sizeof(kFeatures) / sizeof(kFeatures[0]) == kCount,
              "kFeatures must have one entry per CpuFeature");

// ── OS report ────────────────────────────────────────────────────────────────

#if defined(_WIN32)
// The ARM64 kernel publishes the boot CPU's ID registers as REG_QWORD values
// named "CP <hex>" where hex = 0x4000 | CRm << 3 | op2 for op0=3 op1=0 CRn=0
// (MIDR_EL1 is "CP 4000"). This is what .NET and cpuinfo use.
bool win_id_reg(uint8_t reg, uint64_t* out) {
    static const wchar_t* const kNames[kIdCount_] = {
        nullptr, L"CP 4030", L"CP 4031", L"CP 4020" };
    static uint64_t cache[kIdCount_] = {};
    static int8_t   state[kIdCount_] = {};   // 0 unread, 1 ok, -1 failed
    if (reg == kIdNone || reg >= kIdCount_) return false;
    if (state[reg] == 0) {
        DWORD type = 0;
        DWORD size = sizeof(cache[reg]);
        const LSTATUS st = RegGetValueW(
            HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            kNames[reg], RRF_RT_REG_QWORD, &type, &cache[reg], &size);
        state[reg] = (st == ERROR_SUCCESS && size == sizeof(cache[reg])) ? 1 : -1;
    }
    if (state[reg] < 0) return false;
    *out = cache[reg];
    return true;
}
#endif

FeatureReport os_report(const FeatureInfo& f) {
#if defined(__APPLE__)
    char key[64] = "hw.optional.arm.";
    size_t n = 16;
    for (const char* s = f.sysctl; *s && n < sizeof(key) - 1; ++s) key[n++] = *s;
    key[n] = '\0';
    int val = 0;
    size_t len = sizeof(val);
    const bool ok = sysctlbyname(key, &val, &len, nullptr, 0) == 0 && val != 0;
    return ok ? FeatureReport::Present : FeatureReport::Absent;
#elif defined(__linux__)
    bool ok = false;
    if (f.hwcap)       ok = (getauxval(AT_HWCAP)  & f.hwcap)  != 0;
    else if (f.hwcap2) ok = (getauxval(AT_HWCAP2) & f.hwcap2) != 0;
    return ok ? FeatureReport::Present : FeatureReport::Absent;
#elif defined(_WIN32)
    uint64_t reg = 0;
    if (f.id.reg != kIdNone && win_id_reg(f.id.reg, &reg)) {
        const unsigned field = static_cast<unsigned>((reg >> f.id.shift) & 0xF);
        const bool ok = field != 0xF && field >= f.id.min;
        return ok ? FeatureReport::Present : FeatureReport::Absent;
    }
    if (f.win_pf >= 0) {
        const bool ok = IsProcessorFeaturePresent(static_cast<DWORD>(f.win_pf)) != 0;
        return ok ? FeatureReport::Present : FeatureReport::Absent;
    }
    return FeatureReport::Unknown;
#else
    (void)f;
    return FeatureReport::Unknown;
#endif
}

// ── Instruction probe ────────────────────────────────────────────────────────

using ProbeFn = void (*)(void* scratch);

// `insn; ret` in executable memory. A private JitRuntime rather than
// g_jit_pool so detection has no ordering dependence on main().
ProbeFn build_probe(asmjit::JitRuntime& rt, uint32_t insn) {
    asmjit::CodeHolder code;
    if (code.init(rt.environment(), rt.cpu_features()) != asmjit::kErrorOk) return nullptr;
    asmjit::a64::Assembler a(&code);
    const uint32_t words[2] = { insn, 0xD65F03C0u /* ret */ };
    a.embed(words, sizeof(words));
    ProbeFn fn = nullptr;
    if (rt.add(&fn, &code) != asmjit::kErrorOk) return nullptr;
    return fn;
}

#if defined(_WIN32)
// The probe has no unwind info; the unwinder treats a function without a
// .pdata entry as a leaf (pc ← lr, sp unchanged), which it is. Catch every
// code: an undefined encoding is EXCEPTION_ILLEGAL_INSTRUCTION, but a
// feature the OS traps could surface as EXCEPTION_PRIV_INSTRUCTION.
bool run_probe(ProbeFn fn, void* scratch) {
    __try {
        fn(scratch);
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}
void probe_guard_begin() {}
void probe_guard_end()   {}
#else
sigjmp_buf       s_probe_jmp;
struct sigaction s_probe_old_sigill;

void probe_sigill(int) { siglongjmp(s_probe_jmp, 1); }

// sigsetjmp(…, 1) restores the signal mask, so a SIGILL taken inside the
// probe does not leave SIGILL blocked afterwards.
bool run_probe(ProbeFn fn, void* scratch) {
    if (sigsetjmp(s_probe_jmp, 1) != 0) return false;
    fn(scratch);
    return true;
}
void probe_guard_begin() {
    struct sigaction sa = {};
    sa.sa_handler = probe_sigill;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &s_probe_old_sigill);
}
void probe_guard_end() {
    sigaction(SIGILL, &s_probe_old_sigill, nullptr);
}
#endif

// ── Cache ────────────────────────────────────────────────────────────────────

struct FeatureState {
    bool          init = false;
    FeatureReport os[kCount];
    int8_t        probe[kCount];   // 1 executed, 0 faulted, -1 no probe
    bool          has[kCount];
};

FeatureState& state() {
    static FeatureState s;
    if (s.init) return s;
    s.init = true;

    for (size_t i = 0; i < kCount; ++i) s.os[i] = os_report(kFeatures[i]);

    // One handler install, one pass over every probe. The scratch buffer is
    // the memory operand of the load/atomic probes (LDIAPP reads 16 bytes).
    alignas(64) static uint8_t scratch[64] = {};
    {
        asmjit::JitRuntime rt;
        probe_guard_begin();
        for (size_t i = 0; i < kCount; ++i) {
            s.probe[i] = -1;
            if (!kFeatures[i].probe) continue;
            ProbeFn fn = build_probe(rt, kFeatures[i].probe);
            if (!fn) continue;
            s.probe[i] = run_probe(fn, scratch) ? 1 : 0;
            rt.release(fn);
        }
        probe_guard_end();
    }

    for (size_t i = 0; i < kCount; ++i) {
        switch (s.os[i]) {
        case FeatureReport::Present:
            s.has[i] = s.probe[i] != 0;
            if (!s.has[i])
                fprintf(stderr, "warning: the OS reports %s but its instruction raised an "
                                "illegal-instruction trap; treating it as absent\n",
                        kFeatures[i].name);
            break;
        case FeatureReport::Unknown:
            s.has[i] = s.probe[i] == 1;
            break;
        case FeatureReport::Absent:
        default:
            s.has[i] = false;
            break;
        }
    }
    return s;
}

} // namespace

bool cpu_has(CpuFeature feat) {
    const size_t i = static_cast<size_t>(feat);
    if (i >= kCount) return false;
    return state().has[i];
}

FeatureReport cpu_os_reports(CpuFeature feat) {
    const size_t i = static_cast<size_t>(feat);
    if (i >= kCount) return FeatureReport::Unknown;
    return state().os[i];
}

int cpu_probe(CpuFeature feat) {
    const size_t i = static_cast<size_t>(feat);
    if (i >= kCount) return -1;
    return state().probe[i];
}

const char* cpu_feature_name(CpuFeature feat) {
    const size_t i = static_cast<size_t>(feat);
    return (i < kCount) ? kFeatures[i].name : "?";
}

} // namespace arm64bench
