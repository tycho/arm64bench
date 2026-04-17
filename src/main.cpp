// main.cpp
// arm64bench entry point.
//
// Currently a skeleton. As test generators are added, they will be
// called from the appropriate section below.

#include "harness.h"
#include "jit_buffer.h"
#include "timer.h"
#include "cycle_counter.h"
#include <cstdio>
#include <cstring>

#include "gen_integer.h"
#include "gen_memory.h"
#include "gen_branch.h"
#include "gen_fp_simd.h"
#include "gen_pitfalls.h"
#include "gen_lse.h"

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --MHz <n>       Override CPU frequency estimate (MHz)\n");
    printf("  --samples <n>   Samples per benchmark (default 7)\n");
    printf("  --warmup  <n>   Warm-up calls before timing (default 2)\n");
    printf("  --csv           Machine-readable CSV output\n");
    printf("  --all           Run all test categories\n");
    printf("  --integer       Run integer ALU tests\n");
    printf("  --memory        Run memory / cache hierarchy tests\n");
    printf("  --branch        Run branch prediction tests\n");
    printf("  --simd          Run FP / NEON / SVE2 tests\n");
    printf("  --lse           Run LSE atomic RMW tests\n");
    printf("  --pitfalls      Run Apple vs Snapdragon pathology tests\n");
    printf("\n");
}

int main(int argc, char** argv) {
    // ── Parse arguments ────────────────────────────────────────────────────
    bool run_integer  = false;
    bool run_memory   = false;
    bool run_branch   = false;
    bool run_simd     = false;
    bool run_lse      = false;
    bool run_pitfalls = false;
    bool csv_mode     = false;
    uint64_t override_mhz = 0;

    arm64bench::BenchmarkParams default_params{};
    default_params.loops                = 6'000'128;
    default_params.instructions_per_loop = 32;
    // (num_samples, num_warmup, etc. use struct defaults)

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--csv") == 0) {
            csv_mode = true;
        } else if (strcmp(arg, "--all") == 0) {
            run_integer = run_memory = run_branch = run_simd = run_lse = run_pitfalls = true;
        } else if (strcmp(arg, "--integer")  == 0) { run_integer  = true; }
        else if   (strcmp(arg, "--memory")   == 0) { run_memory   = true; }
        else if   (strcmp(arg, "--branch")   == 0) { run_branch   = true; }
        else if   (strcmp(arg, "--simd")     == 0) { run_simd     = true; }
        else if   (strcmp(arg, "--lse")      == 0) { run_lse      = true; }
        else if   (strcmp(arg, "--pitfalls") == 0) { run_pitfalls = true; }
        else if (strcmp(arg, "--MHz") == 0 && i + 1 < argc) {
            override_mhz = static_cast<uint64_t>(atoll(argv[++i]));
        } else if (strcmp(arg, "--samples") == 0 && i + 1 < argc) {
            default_params.num_samples = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            default_params.num_warmup = static_cast<uint32_t>(atoi(argv[++i]));
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Default: run integer and memory tests if nothing specified.
    if (!run_integer && !run_memory && !run_branch && !run_simd && !run_lse && !run_pitfalls)
        run_integer = run_memory = true;

    // ── Initialise ─────────────────────────────────────────────────────────
    printf("\narm64bench  (built %s %s)\n", __DATE__, __TIME__);

    // Initialise the process-wide JIT pool before anything tries to compile.
    arm64bench::JitPool jit_pool;
    arm64bench::g_jit_pool = &jit_pool;

    // Set output mode before any benchmark() calls print anything.
    if (csv_mode) {
        arm64bench::set_output_mode(arm64bench::OutputMode::CSV);
        arm64bench::print_csv_header();
    }

    // CPU frequency: prefer explicit override, otherwise calibrate.
    if (override_mhz > 0) {
        arm64bench::g_cpu_freq_hz = override_mhz * 1'000'000ULL;
        printf("CPU frequency: %llu MHz (user override)\n",
               static_cast<unsigned long long>(override_mhz));
    } else {
        printf("Calibrating CPU frequency...\n");
        const uint64_t hz = arm64bench::calibrate_cpu_freq();
        if (hz > 0) {
            printf("CPU frequency: ~%llu MHz (calibrated)\n",
                   static_cast<unsigned long long>(hz / 1'000'000ULL));
        } else {
            printf("CPU frequency: unknown (clock cycle counts will be suppressed)\n"
                   "  Pass --MHz <n> to provide it manually.\n");
        }
    }

    // Initialise hardware PMU cycle counters (Tier 1).
    const bool pmu_ok = arm64bench::cycle_counter_init();

    // Build the Tier 2 ratio-normalization reference function (ADD latency chain).
    // Always created so it is available as fallback when PMU is unavailable.
    // 500 000 iterations × 32 unroll = 16 M ADD instructions per call (~5 ms at 3 GHz).
    {
        static constexpr uint64_t kRefLoops  = 500'000;
        static constexpr uint32_t kRefUnroll = 32;
        arm64bench::ReferenceParams ref;
        ref.fn          = arm64bench::gen::create_add_latency_ref(kRefLoops, kRefUnroll);
        ref.total_insns = kRefLoops * kRefUnroll;
        arm64bench::set_reference_function(ref);
    }

    if (pmu_ok) {
        printf("CPU cycle source: hardware PMU (Tier 1 — P-state immune)\n\n");
    } else {
        printf("CPU cycle source: ratio normalization vs ADD reference"
               " (Tier 2 — drift-resistant)\n\n");
    }

    // ── Run selected test categories ────────────────────────────────────────
    // Each generator creates, runs, and releases its test functions.
    // Generators are not yet implemented; these are the intended call sites.

    if (run_integer) {
        printf("── Integer ALU tests ─────────────────────────────────────────\n");
        arm64bench::gen::run_integer_tests(default_params);
    }

    if (run_memory) {
        printf("── Memory / cache hierarchy tests ────────────────────────────\n");
        arm64bench::gen::run_memory_tests(default_params);
        printf("\n");
    }

    if (run_branch) {
        printf("── Branch prediction tests ───────────────────────────────────\n");
        arm64bench::gen::run_branch_tests(default_params);
        printf("\n");
    }

    if (run_simd) {
        printf("── FP / NEON / SVE2 tests ────────────────────────────────────\n");
        arm64bench::gen::run_fp_simd_tests(default_params);
        printf("\n");
    }

    if (run_lse) {
        printf("── LSE atomic tests ──────────────────────────────────────────\n");
        arm64bench::gen::run_lse_tests(default_params);
        printf("\n");
    }

    if (run_pitfalls) {
        printf("── Apple vs Snapdragon pathology tests ───────────────────────\n");
        arm64bench::gen::run_pitfall_tests(default_params);
        printf("\n");
    }

    // g_jit_pool goes out of scope here, releasing all compiled functions.
    arm64bench::g_jit_pool = nullptr;
    return 0;
}
