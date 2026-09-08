// affinity.cpp
// See affinity.h.

#include "affinity.h"

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

#if defined(_WIN32)

bool affinity_supported() { return true; }

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        if (!(proc & (DWORD_PTR(1) << i))) continue;
        if (n < max) out[n] = i;
        ++n;
    }
    return n;
}

bool pin_thread_to_cpu(uint32_t cpu) {
    if (cpu >= 64) return false;
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) != 0;
}

void unpin_thread() {
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys))
        SetThreadAffinityMask(GetCurrentThread(), proc);
}

void set_thread_cluster_hint(bool) {}

#elif defined(__linux__)

bool affinity_supported() { return true; }

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &set)) continue;
        if (n < max) out[n] = i;
        ++n;
    }
    return n;
}

bool pin_thread_to_cpu(uint32_t cpu) {
    if (cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;   // pid 0 = calling thread
}

void unpin_thread() {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (uint32_t i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i, &set);
    sched_setaffinity(0, sizeof(set), &set);   // the kernel clips to the allowed set
}

void set_thread_cluster_hint(bool) {}

#else   // macOS

bool affinity_supported() { return false; }

uint32_t allowed_cpus(uint32_t* out, uint32_t max) {
    const uint32_t n = std::thread::hardware_concurrency();
    for (uint32_t i = 0; i < n && i < max; ++i) out[i] = i;
    return n;
}

bool pin_thread_to_cpu(uint32_t) { return false; }
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
