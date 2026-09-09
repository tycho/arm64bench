// cpu_select.cpp
// See cpu_select.h.

#include "cpu_select.h"

#include "gen_integer.h"
#include "harness.h"
#include "jit_buffer.h"
#include "timer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace arm64bench {

namespace {

// ── Clock survey ─────────────────────────────────────────────────────────────

// Dependent ADD chain: one instruction per cycle on every ARM core, so
// instructions per nanosecond is the clock in GHz. A few warm calls let the
// governor ramp the core before the timed ones.
double survey_ghz(uint32_t cpu, JitPool::TestFn fn, double insns) {
    if (!pin_thread_to_cpu(cpu)) return 0.0;
    for (int i = 0; i < 4; ++i) fn();
    double best_ns = 1e300;
    for (int i = 0; i < 8; ++i) {
        const RawTick t0 = tick_now();
        fn();
        const double ns = ticks_to_ns_f(tick_now() - t0);
        if (ns < best_ns) best_ns = ns;
    }
    return best_ns > 0.0 ? insns / best_ns : 0.0;
}

// ── Formatting ───────────────────────────────────────────────────────────────

// "0-5" / "0-3,8" style list of the ids selected by `in`.
void format_ids(char* out, size_t cap, const CpuTopology& t, const bool* in) {
    size_t n = 0;
    out[0] = '\0';
    for (uint32_t i = 0; i < t.count;) {
        if (!in[i]) { ++i; continue; }
        uint32_t j = i;
        while (j + 1 < t.count && in[j + 1] && t.cpus[j + 1].id == t.cpus[j].id + 1) ++j;
        if (j > i) n += static_cast<size_t>(snprintf(out + n, cap - n, "%s%u-%u", n ? "," : "", t.cpus[i].id, t.cpus[j].id));
        else       n += static_cast<size_t>(snprintf(out + n, cap - n, "%s%u", n ? "," : "", t.cpus[i].id));
        if (n >= cap) break;
        i = j + 1;
    }
}

// Core name from MIDR_EL1 (implementer[31:24], variant[23:20], part[15:4],
// revision[3:0]), with the variant and revision in Arm's rNpM form: the
// X2 Elite reports its Performance cores as r2p1 and its Prime cores as
// r1p1 of the same part.
const char* core_name(uint64_t midr, char* buf, size_t cap) {
    const unsigned impl = static_cast<unsigned>((midr >> 24) & 0xFF);
    const unsigned var  = static_cast<unsigned>((midr >> 20) & 0xF);
    const unsigned part = static_cast<unsigned>((midr >> 4) & 0xFFF);
    const unsigned rev  = static_cast<unsigned>(midr & 0xF);
    const char* name = nullptr;
    if (impl == 0x41) {
        switch (part) {
            case 0xD03: name = "Cortex-A53"; break;  case 0xD05: name = "Cortex-A55"; break;
            case 0xD07: name = "Cortex-A57"; break;  case 0xD08: name = "Cortex-A72"; break;
            case 0xD09: name = "Cortex-A73"; break;  case 0xD0A: name = "Cortex-A75"; break;
            case 0xD0B: name = "Cortex-A76"; break;  case 0xD0C: name = "Neoverse-N1"; break;
            case 0xD0D: name = "Cortex-A77"; break;  case 0xD40: name = "Neoverse-V1"; break;
            case 0xD41: name = "Cortex-A78"; break;  case 0xD44: name = "Cortex-X1"; break;
            case 0xD46: name = "Cortex-A510"; break; case 0xD47: name = "Cortex-A710"; break;
            case 0xD48: name = "Cortex-X2"; break;   case 0xD49: name = "Neoverse-N2"; break;
            case 0xD4B: name = "Cortex-A78C"; break; case 0xD4C: name = "Cortex-X1C"; break;
            case 0xD4D: name = "Cortex-A715"; break; case 0xD4E: name = "Cortex-X3"; break;
            case 0xD4F: name = "Neoverse-V2"; break; case 0xD80: name = "Cortex-A520"; break;
            case 0xD81: name = "Cortex-A720"; break; case 0xD82: name = "Cortex-X4"; break;
            case 0xD84: name = "Neoverse-V3"; break; case 0xD85: name = "Cortex-X925"; break;
            case 0xD87: name = "Cortex-A725"; break; case 0xD8E: name = "Neoverse-N3"; break;
            default: break;
        }
    } else if (impl == 0x51) {
        // 0x001 is QCOM_CPU_PART_ORYON_X1 in the Linux kernel's cputype.h (the
        // X1E80100 "X Elite" generation); 0x002 was read off an X2E96100.
        if (part == 0x001)      name = "Oryon (Snapdragon X1)";
        else if (part == 0x002) name = "Oryon (Snapdragon X2)";
        else if (part == 0x804) name = "Kryo 4xx Gold";
        else if (part == 0x805) name = "Kryo 4xx Silver";
    } else if (impl == 0xC0 && part == 0xAC3) {
        name = "Ampere-1";
    }
    if (name) snprintf(buf, cap, "%s r%up%u", name, var, rev);
    else      snprintf(buf, cap, "impl 0x%02x part 0x%03x r%up%u", impl, part, var, rev);
    return buf;
}

} // namespace

bool parse_cpu_arg(const char* s, int* mode) {
    if (strcmp(s, "auto") == 0) { *mode = kCpuAuto; return true; }
    if (strcmp(s, "any") == 0)  { *mode = kCpuAny;  return true; }
    char* end = nullptr;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || v < 0 || v >= static_cast<long>(kMaxCpus)) return false;
    *mode = static_cast<int>(v);
    return true;
}

void select_cpus(int mode, bool survey, CpuChoice& out) {
    out.n = 0;
    out.pinned = false;

    CpuTopology t;
    cpu_topology(t);
    if (t.count == 0) {
        printf("CPU topology: unknown\n");
        return;
    }

    if (!affinity_supported()) {
        printf("CPU topology: %u CPUs; no thread affinity on this OS, the main thread runs at\n"
               "  user-interactive QoS, which keeps it on the performance cluster\n", t.count);
        return;
    }

    // Measured clock per CPU.
    double ghz[kMaxCpus] = {};
    bool surveyed = false;
    if (survey && g_jit_pool) {
        const uint64_t loops  = scale_loops(200'000);
        const uint32_t unroll = 32;
        if (JitPool::TestFn fn = gen::create_add_latency_ref(loops, unroll)) {
            uint32_t mine[kMaxCpus];
            uint32_t nm = thread_cpus(mine, kMaxCpus);
            if (nm > kMaxCpus) nm = kMaxCpus;
            for (uint32_t i = 0; i < t.count; ++i)
                ghz[i] = survey_ghz(t.cpus[i].id, fn, static_cast<double>(loops) * unroll);
            if (nm) pin_thread_to_cpus(mine, nm); else unpin_thread();
            g_jit_pool->release(fn);
            surveyed = true;
        }
    }

    // Group into clusters: the OS's L2 groups, else CPUs of one class whose
    // measured clocks agree within 3 %, else each CPU alone.
    int32_t group[kMaxCpus];
    int32_t ngroups = 0;
    for (uint32_t i = 0; i < t.count; ++i) group[i] = -1;
    for (uint32_t i = 0; i < t.count; ++i) {
        if (group[i] >= 0) continue;
        group[i] = ngroups;
        for (uint32_t j = i + 1; j < t.count; ++j) {
            if (group[j] >= 0) continue;
            bool same;
            if (t.have_clusters)
                same = t.cpus[j].cluster == t.cpus[i].cluster && t.cpus[i].cluster >= 0;
            else if (surveyed)
                same = t.cpus[j].perf_class == t.cpus[i].perf_class && ghz[i] > 0.0 &&
                       ghz[j] > ghz[i] * 0.97 && ghz[j] < ghz[i] * 1.03;
            else
                same = t.have_classes && t.cpus[j].perf_class == t.cpus[i].perf_class;
            if (same) group[j] = ngroups;
        }
        ++ngroups;
    }

    // Per-group score: median measured clock, else OS max clock, else class.
    double  score[kMaxCpus] = {};
    int32_t gclass[kMaxCpus];
    uint32_t gfirst[kMaxCpus];
    for (int32_t g = 0; g < ngroups; ++g) {
        double vals[kMaxCpus];
        uint32_t nv = 0;
        gclass[g] = -1;
        gfirst[g] = UINT32_MAX;
        for (uint32_t i = 0; i < t.count; ++i) {
            if (group[i] != g) continue;
            if (surveyed && ghz[i] > 0.0) vals[nv++] = ghz[i];
            if (t.cpus[i].perf_class > gclass[g]) gclass[g] = t.cpus[i].perf_class;
            if (t.cpus[i].id < gfirst[g]) gfirst[g] = t.cpus[i].id;
        }
        if (nv) {
            for (uint32_t a = 1; a < nv; ++a)   // insertion sort, tiny n
                for (uint32_t b = a; b > 0 && vals[b - 1] > vals[b]; --b) {
                    const double tmp = vals[b]; vals[b] = vals[b - 1]; vals[b - 1] = tmp;
                }
            score[g] = vals[nv / 2];
        } else {
            uint32_t khz = 0;
            for (uint32_t i = 0; i < t.count; ++i)
                if (group[i] == g && t.cpus[i].max_khz > khz) khz = t.cpus[i].max_khz;
            score[g] = khz ? khz / 1e6 : (gclass[g] >= 0 ? gclass[g] : 0.0);
        }
    }

    // Print the table.
    printf("CPU topology: %u CPUs in %d %s%s\n", t.count, ngroups,
           t.have_clusters ? "L2 cluster" : "group", ngroups == 1 ? "" : "s");
    if (!t.have_clusters && ngroups > 1) printf("  (L2 sharing unknown; grouped by class and measured clock)\n");
    for (int32_t g = 0; g < ngroups; ++g) {
        bool in[kMaxCpus];
        for (uint32_t i = 0; i < t.count; ++i) in[i] = group[i] == g;
        char ids[128];
        format_ids(ids, sizeof(ids), t, in);
        uint32_t khz = 0; uint64_t midr = 0;
        for (uint32_t i = 0; i < t.count; ++i)
            if (group[i] == g) { if (t.cpus[i].max_khz > khz) khz = t.cpus[i].max_khz; if (!midr) midr = t.cpus[i].midr; }
        char name[40];
        printf("  cpus %-8s  %-22s", ids, midr ? core_name(midr, name, sizeof(name)) : "");
        if (gclass[g] >= 0) printf("  class %d", gclass[g]); else printf("         ");
        if (surveyed && score[g] > 0.0) printf("  measured %.2f GHz", score[g]);
        // The OS figure is the registry's "~MHz" on Windows (wrong on the 8cx
        // Gen 3: 1.37 GHz for cores that run at 2.99) or cpufreq's max on
        // Linux; shown for the record, never used when the survey ran.
        if (khz)  printf("  (OS says %.2f GHz)", khz / 1e6);
        if (midr) printf("  MIDR 0x%08llx", static_cast<unsigned long long>(midr));
        printf("\n");
    }

    if (mode == kCpuAny) {
        printf("Main thread: not pinned (--cpu any)\n");
        return;
    }

    // Choose.
    int32_t pick = -1;
    const char* why = "";
    if (mode >= 0) {
        for (uint32_t i = 0; i < t.count; ++i)
            if (t.cpus[i].id == static_cast<uint32_t>(mode)) { pick = group[i]; break; }
        if (pick < 0) {
            printf("Main thread: cpu %d is not available to this process; not pinned\n", mode);
            return;
        }
        why = "requested with --cpu";
    } else {
        for (int32_t g = 0; g < ngroups; ++g) {
            if (pick < 0) { pick = g; continue; }
            const bool faster = score[g] > score[pick] * 1.03;
            const bool tie    = !faster && score[g] > score[pick] * 0.97;
            if (faster || (tie && (gclass[g] > gclass[pick] ||
                                   (gclass[g] == gclass[pick] && gfirst[g] < gfirst[pick]))))
                pick = g;
        }
        why = surveyed ? "fastest measured clock" : (t.have_classes ? "highest OS class" : "first group");
    }

    out.n = 0;
    bool in[kMaxCpus];
    for (uint32_t i = 0; i < t.count; ++i) {
        in[i] = group[i] == pick;
        if (in[i]) out.cpus[out.n++] = t.cpus[i].id;
    }
    char ids[128];
    format_ids(ids, sizeof(ids), t, in);
    out.pinned = pin_thread_to_cpus(out.cpus, out.n);
    if (out.pinned)
        printf("Main thread: pinned to cpus %s (%s; --cpu N picks the cluster holding cpu N, --cpu any floats)\n",
               ids, why);
    else
        printf("Main thread: pinning to cpus %s failed; not pinned\n", ids);
}

} // namespace arm64bench
