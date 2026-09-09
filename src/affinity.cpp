// affinity.cpp
// See affinity.h.

#include "affinity.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#elif defined(__APPLE__)
#  include <pthread.h>
#  include <sys/qos.h>
#endif

namespace arm64bench {

namespace {

void reset_topology(CpuTopology& t) {
    uint32_t ids[kMaxCpus];
    uint32_t n = allowed_cpus(ids, kMaxCpus);
    if (n > kMaxCpus) n = kMaxCpus;
    t.count = n;
    t.have_clusters = false;
    t.have_classes = false;
    for (uint32_t i = 0; i < n; ++i)
        t.cpus[i] = CpuInfo{ ids[i], -1, -1, 0, 0 };
}

[[maybe_unused]] CpuInfo* find_cpu(CpuTopology& t, uint32_t id) {
    for (uint32_t i = 0; i < t.count; ++i)
        if (t.cpus[i].id == id) return &t.cpus[i];
    return nullptr;
}

} // namespace

#if defined(_WIN32)

bool affinity_supported() { return true; }

static uint32_t mask_to_list(DWORD_PTR mask, uint32_t* out, uint32_t max) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        if (!(mask & (DWORD_PTR(1) << i))) continue;
        if (n < max) out[n] = i;
        ++n;
    }
    return n;
}

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    static DWORD_PTR s_mask = 0;
    if (!s_mask) {
        DWORD_PTR proc = 0, sys = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) return 0;
        s_mask = proc;
    }
    return mask_to_list(s_mask, out, max);
}

uint32_t thread_cpus(uint32_t* out, uint32_t max) {
    GROUP_AFFINITY ga = {};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga)) return 0;
    return mask_to_list(ga.Mask, out, max);
}

bool pin_thread_to_cpus(const uint32_t* cpus, uint32_t n) {
    DWORD_PTR mask = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (cpus[i] >= 64) return false;
        mask |= DWORD_PTR(1) << cpus[i];
    }
    if (!mask) return false;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

bool pin_thread_to_cpu(uint32_t cpu) { return pin_thread_to_cpus(&cpu, 1); }

void unpin_thread() {
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys))
        SetThreadAffinityMask(GetCurrentThread(), proc);
}

void set_thread_cluster_hint(bool) {}

// Windows publishes the core/cache relationships from the firmware's PPTT
// table, with EfficiencyClass distinguishing big and little cores (higher
// is faster). Per-CPU nominal MHz and MIDR_EL1 sit in the registry under
// CentralProcessor\<n> as "~MHz" and "CP 4000".
bool cpu_topology(CpuTopology& t) {
    reset_topology(t);

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    if (len) {
        uint8_t* buf = static_cast<uint8_t*>(malloc(len));
        if (buf && GetLogicalProcessorInformationEx(
                RelationAll, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf), &len)) {
            int32_t next_cluster = 0;
            for (DWORD off = 0; off < len;) {
                auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf + off);
                if (info->Size == 0) break;
                if (info->Relationship == RelationProcessorCore) {
                    const GROUP_AFFINITY& g = info->Processor.GroupMask[0];
                    if (g.Group == 0) {
                        for (uint32_t i = 0; i < t.count; ++i) {
                            const uint32_t id = t.cpus[i].id;
                            if (id < 64 && (g.Mask & (KAFFINITY(1) << id))) {
                                t.cpus[i].perf_class = info->Processor.EfficiencyClass;
                                t.have_classes = true;
                            }
                        }
                    }
                } else if (info->Relationship == RelationCache && info->Cache.Level == 2 &&
                           (info->Cache.Type == CacheUnified || info->Cache.Type == CacheData)) {
                    const GROUP_AFFINITY& g = info->Cache.GroupMask;
                    if (g.Group == 0) {
                        bool used = false;
                        for (uint32_t i = 0; i < t.count; ++i) {
                            const uint32_t id = t.cpus[i].id;
                            if (id < 64 && (g.Mask & (KAFFINITY(1) << id)) && t.cpus[i].cluster < 0) {
                                t.cpus[i].cluster = next_cluster;
                                used = true;
                            }
                        }
                        if (used) { ++next_cluster; t.have_clusters = true; }
                    }
                }
                off += info->Size;
            }
        }
        free(buf);
    }

    for (uint32_t i = 0; i < t.count; ++i) {
        wchar_t key[80];
        swprintf(key, 80, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\%u", t.cpus[i].id);
        DWORD mhz = 0, size = sizeof(mhz);
        if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"~MHz", RRF_RT_REG_DWORD, nullptr, &mhz, &size)
                == ERROR_SUCCESS)
            t.cpus[i].max_khz = mhz * 1000u;
        uint64_t midr = 0;
        size = sizeof(midr);
        if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"CP 4000", RRF_RT_REG_QWORD, nullptr, &midr, &size)
                == ERROR_SUCCESS)
            t.cpus[i].midr = midr;
    }
    return true;
}

#elif defined(__linux__)

bool affinity_supported() { return true; }

static uint32_t set_to_list(const cpu_set_t& set, uint32_t* out, uint32_t max) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &set)) continue;
        if (n < max) out[n] = i;
        ++n;
    }
    return n;
}

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    static cpu_set_t s_set;
    static bool      s_have = false;
    if (!s_have) {
        CPU_ZERO(&s_set);
        if (sched_getaffinity(0, sizeof(s_set), &s_set) != 0) return 0;
        s_have = true;
    }
    return set_to_list(s_set, out, max);
}

uint32_t thread_cpus(uint32_t* out, uint32_t max) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;   // pid 0 = calling thread
    return set_to_list(set, out, max);
}

bool pin_thread_to_cpus(const uint32_t* cpus, uint32_t n) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (uint32_t i = 0; i < n; ++i) {
        if (cpus[i] >= CPU_SETSIZE) return false;
        CPU_SET(cpus[i], &set);
    }
    if (!n) return false;
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

bool pin_thread_to_cpu(uint32_t cpu) { return pin_thread_to_cpus(&cpu, 1); }

void unpin_thread() {
    uint32_t ids[kMaxCpus];
    uint32_t n = allowed_cpus(ids, kMaxCpus);
    if (n > kMaxCpus) n = kMaxCpus;
    pin_thread_to_cpus(ids, n);
}

void set_thread_cluster_hint(bool) {}

// sysfs: cache/indexN/{level,type,shared_cpu_list} for the L2 group,
// cpu_capacity (the scheduler's relative capacity, 1024 = biggest core),
// cpufreq/cpuinfo_max_freq (kHz), regs/identification/midr_el1.
static bool read_text(const char* path, char* out, size_t cap) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    size_t n = fread(out, 1, cap - 1, f);
    fclose(f);
    out[n] = '\0';
    while (n && (out[n - 1] == '\n' || out[n - 1] == ' ')) out[--n] = '\0';
    return n > 0;
}

static bool read_u64(const char* path, uint64_t* v, int base) {
    char buf[64];
    if (!read_text(path, buf, sizeof(buf))) return false;
    char* end = nullptr;
    *v = strtoull(buf, &end, base);
    return end != buf;
}

// Apply `fn(cpu)` to every id in a "0-3,8,10-11" list.
template <typename F>
static void for_each_in_list(const char* list, F&& fn) {
    const char* p = list;
    while (*p) {
        char* end = nullptr;
        const unsigned long a = strtoul(p, &end, 10);
        if (end == p) break;
        unsigned long b = a;
        p = end;
        if (*p == '-') { b = strtoul(p + 1, &end, 10); p = end; }
        for (unsigned long c = a; c <= b && c < kMaxCpus; ++c) fn(static_cast<uint32_t>(c));
        while (*p == ',' || *p == ' ') ++p;
    }
}

bool cpu_topology(CpuTopology& t) {
    reset_topology(t);
    char path[128], text[512];

    // Clusters from L2 sharing.
    int32_t next_cluster = 0;
    for (uint32_t i = 0; i < t.count; ++i) {
        if (t.cpus[i].cluster >= 0) continue;
        const uint32_t id = t.cpus[i].id;
        for (int k = 0; k < 8; ++k) {
            uint64_t level = 0;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%d/level", id, k);
            if (!read_u64(path, &level, 10)) break;
            if (level != 2) continue;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%d/type", id, k);
            if (read_text(path, text, sizeof(text)) && strcmp(text, "Instruction") == 0) continue;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cache/index%d/shared_cpu_list", id, k);
            if (!read_text(path, text, sizeof(text))) break;
            bool used = false;
            for_each_in_list(text, [&](uint32_t c) {
                if (CpuInfo* ci = find_cpu(t, c); ci && ci->cluster < 0) { ci->cluster = next_cluster; used = true; }
            });
            if (used) { ++next_cluster; t.have_clusters = true; }
            break;
        }
    }

    // Classes from cpu_capacity: rank the distinct values.
    uint64_t caps[kMaxCpus];
    bool have_cap = false;
    for (uint32_t i = 0; i < t.count; ++i) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpu_capacity", t.cpus[i].id);
        caps[i] = 0;
        if (read_u64(path, &caps[i], 10)) have_cap = true;
    }
    if (have_cap) {
        for (uint32_t i = 0; i < t.count; ++i) {
            int32_t rank = 0;
            for (uint32_t j = 0; j < t.count; ++j) {
                if (caps[j] < caps[i]) {
                    bool counted = false;
                    for (uint32_t k = 0; k < j; ++k) if (caps[k] == caps[j]) { counted = true; break; }
                    if (!counted) ++rank;
                }
            }
            t.cpus[i].perf_class = rank;
        }
        t.have_classes = true;
    }

    for (uint32_t i = 0; i < t.count; ++i) {
        uint64_t v = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", t.cpus[i].id);
        if (read_u64(path, &v, 10)) t.cpus[i].max_khz = static_cast<uint32_t>(v);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/regs/identification/midr_el1", t.cpus[i].id);
        if (read_u64(path, &v, 16)) t.cpus[i].midr = v;
    }
    return true;
}

#else   // macOS

bool affinity_supported() { return false; }

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    const uint32_t n = std::thread::hardware_concurrency();
    for (uint32_t i = 0; i < n && i < max; ++i) out[i] = i;
    return n;
}

uint32_t thread_cpus(uint32_t* out, uint32_t max) { return allowed_cpus(out, max); }

bool cpu_topology(CpuTopology& t) {
    reset_topology(t);
    return false;
}

bool pin_thread_to_cpu(uint32_t) { return false; }
bool pin_thread_to_cpus(const uint32_t*, uint32_t) { return false; }
void unpin_thread() {}

void set_thread_cluster_hint(bool efficiency) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(efficiency ? QOS_CLASS_BACKGROUND
                                             : QOS_CLASS_USER_INTERACTIVE, 0);
#else
    (void)efficiency;
#endif
}

#endif

} // namespace arm64bench
