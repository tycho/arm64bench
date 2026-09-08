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
#if defined(_WIN32)
#include <Windows.h>
#endif

#include "gen_integer.h"
#include "gen_memory.h"
#include "gen_branch.h"
#include "gen_fp_simd.h"
#include "gen_pitfalls.h"
#include "gen_ooo.h"
#include "gen_sve.h"
#include "gen_mlp.h"
#include "gen_frontend.h"
#include "gen_icache.h"
#include "gen_prefetch.h"
#include "gen_lse.h"

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --MHz <n>       Override CPU frequency estimate (MHz)\n");
    printf("  --samples <n>   Samples per benchmark (default 7)\n");
    printf("  --warmup  <n>   Warm-up calls before timing (default 2)\n");
    printf("  --csv           Machine-readable CSV output\n");
    printf("  --smoke         Execute every test once with tiny loop counts;\n");
    printf("                  report ok/crash, record no measurements (CI)\n");
    printf("  --filter <s>    Only run tests whose name contains <s>\n");
    printf("  --all           Run all test categories\n");
    printf("  --integer       Run integer ALU tests\n");
    printf("  --memory        Run memory / cache hierarchy tests\n");
    printf("  --branch        Run branch prediction tests\n");
    printf("  --simd          Run FP / NEON / SVE2 tests\n");
    printf("  --lse           Run LSE atomic RMW tests\n");
    printf("  --pitfalls      Run Apple vs Snapdragon pathology tests\n");
    printf("  --ooo           Run out-of-order window (ROB / PRF / LSQ) tests\n");
    printf("  --sve           Run SVE/SVE2 tests (native, or streaming via SME)\n");
    printf("  --mlp           Run memory-level parallelism (outstanding-miss) tests\n");
    printf("  --frontend      Run decode width / MOV elimination / zero idiom / fusion tests\n");
    printf("  --icache        Run I-cache size and iTLB reach sweeps\n");
    printf("  --prefetch      Run hardware-prefetcher stride sweeps and PRFM lookahead tests\n");
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
    bool run_ooo      = false;
    bool run_sve      = false;
    bool run_mlp      = false;
    bool run_frontend = false;
    bool run_icache   = false;
    bool run_prefetch = false;
    bool csv_mode     = false;
    bool smoke_mode   = false;
    const char* name_filter = nullptr;
    uint64_t override_mhz = 0;

    arm64bench::BenchmarkParams default_params{};
    default_params.instructions_per_loop = 32;
    // (num_samples, num_warmup, etc. use struct defaults; loops is set below
    //  once the run mode is known)

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--csv") == 0) {
            csv_mode = true;
        } else if (strcmp(arg, "--smoke") == 0) {
            smoke_mode = true;
        } else if (strcmp(arg, "--filter") == 0 && i + 1 < argc) {
            name_filter = argv[++i];
        } else if (strcmp(arg, "--all") == 0) {
            run_integer = run_memory = run_branch = run_simd = run_lse = run_pitfalls = run_ooo = run_sve = run_mlp = run_frontend = run_icache = run_prefetch = true;
        } else if (strcmp(arg, "--integer")  == 0) { run_integer  = true; }
        else if   (strcmp(arg, "--memory")   == 0) { run_memory   = true; }
        else if   (strcmp(arg, "--branch")   == 0) { run_branch   = true; }
        else if   (strcmp(arg, "--simd")     == 0) { run_simd     = true; }
        else if   (strcmp(arg, "--lse")      == 0) { run_lse      = true; }
        else if   (strcmp(arg, "--pitfalls") == 0) { run_pitfalls = true; }
        else if   (strcmp(arg, "--ooo")      == 0) { run_ooo      = true; }
        else if   (strcmp(arg, "--sve")      == 0) { run_sve      = true; }
        else if   (strcmp(arg, "--mlp")      == 0) { run_mlp      = true; }
        else if   (strcmp(arg, "--frontend") == 0) { run_frontend = true; }
        else if   (strcmp(arg, "--icache")   == 0) { run_icache   = true; }
        else if   (strcmp(arg, "--prefetch") == 0) { run_prefetch = true; }
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

#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif

    // Default: run integer and memory tests if nothing specified.
    if (!run_integer && !run_memory && !run_branch && !run_simd && !run_lse && !run_pitfalls && !run_ooo && !run_sve && !run_mlp && !run_frontend && !run_icache && !run_prefetch)
        run_integer = run_memory = true;

    // Run mode must be set before any loop count is derived: scale_loops()
    // consults it, and generators bake the result into their JIT code.
    if (smoke_mode)
        arm64bench::set_run_mode(arm64bench::RunMode::Smoke);
    arm64bench::set_name_filter(name_filter);

    // Nominal 6M iterations × 32 instructions ≈ 60 ms per call at 3 GHz.
    default_params.loops = arm64bench::scale_loops(6'000'128);

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
    } else if (smoke_mode) {
        // ~1.5 s of calibration would produce a number nothing consumes.
        printf("CPU frequency: not calibrated (smoke mode)\n");
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

    if (smoke_mode) {
        printf("Run mode: smoke — each test executed once with loop counts / %llu;"
               " no measurements recorded\n",
               static_cast<unsigned long long>(arm64bench::kSmokeLoopDivisor));
        printf("CPU cycle source: %s (unused in smoke mode)\n\n",
               pmu_ok ? "hardware PMU available" : "hardware PMU unavailable");
    } else if (pmu_ok) {
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

    if (run_ooo) {
        printf("── Out-of-order window tests ─────────────────────────────────\n");
        arm64bench::gen::run_ooo_tests(default_params);
        printf("\n");
    }

    if (run_sve) {
        printf("── SVE / SVE2 tests ─────────────────────────────────────────\n");
        arm64bench::gen::run_sve_tests(default_params);
        printf("\n");
    }

    if (run_mlp) {
        printf("── Memory-level parallelism tests ────────────────────────────\n");
        arm64bench::gen::run_mlp_tests(default_params);
        printf("\n");
    }

    if (run_frontend) {
        printf("── Front-end / rename tests ──────────────────────────────────\n");
        arm64bench::gen::run_frontend_tests(default_params);
        printf("\n");
    }

    if (run_icache) {
        printf("── Instruction cache / iTLB tests ────────────────────────────\n");
        arm64bench::gen::run_icache_tests(default_params);
        printf("\n");
    }

    if (run_prefetch) {
        printf("── Prefetcher tests ──────────────────────────────────────────\n");
        arm64bench::gen::run_prefetch_tests(default_params);
        printf("\n");
    }

    int exit_code = 0;
    if (smoke_mode) {
        const uint32_t n    = arm64bench::smoke_test_count();
        const uint32_t errs = arm64bench::jit_error_count();
        printf("── Smoke summary ─────────────────────────────────────────────\n");
        if (n == 0) {
            printf("FAIL: no tests executed%s\n",
                   name_filter ? " (filter matched nothing)" : "");
            exit_code = 1;
        } else if (errs > 0) {
            // A rejected instruction is a test that ran without part of its
            // body; the run is not evidence that the body encodes.
            printf("FAIL: %u tests executed, but AsmJit rejected %u instruction(s)"
                   " (see 'asmjit error' lines above)\n", n, errs);
            exit_code = 1;
        } else {
            printf("OK: %u tests executed, no encoding errors\n", n);
        }
    }

    // g_jit_pool goes out of scope here, releasing all compiled functions.
    arm64bench::g_jit_pool = nullptr;
    return exit_code;
}
