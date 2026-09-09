# arm64bench

Precision microarchitecture benchmarking framework for ARM64/AArch64 processors. Measures CPU performance across integer ALU, memory hierarchy, branch prediction, and FP/NEON SIMD. Targets Apple Silicon (macOS), Windows ARM64, and Linux AArch64.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
ninja
```

Or with presets (`macos-release`, `linux-clang-release`, `win-clangcl-release`, ...; see `CMakePresets.json`):

```bash
cmake --preset macos-release && cmake --build --preset macos-release && ctest --preset macos-release
```

**Requirements:** CMake 3.25+, Clang (MSVC is explicitly rejected; clang-cl is fine). AsmJit is a
git submodule under `contrib/asmjit` — clone with `--recurse-submodules`.

Default build type is `RelWithDebInfo`. Release builds use `-O3`.

Targets: `arm64bench` (the benchmark), `arm64bench_core` (static library with everything but
`main()`), `arm64bench_selftest` (measurement-machinery tests). CTest registers `selftest` and
`smoke` (= `arm64bench --all --smoke`).

Run with `sudo ./arm64bench` on macOS 15+ (Sequoia/Tahoe) to enable hardware PMU cycle counting; unprivileged runs fall back to Tier 2 ratio normalization.

## Run

```bash
./arm64bench [--all | --integer | --memory | --branch | --simd | --lse | --pitfalls | --ooo | --sve | --mlp | --frontend | --icache | --prefetch | --c2c]
             [--MHz <freq>] [--samples <n>] [--warmup <n>] [--csv]
             [--smoke] [--filter <substr>] [--cpu auto|any|<n>]
```

Default (no flags): runs integer and memory tests.

- `--smoke`: execute every generated test function once with loop counts divided by 1000, print
  `ok` per test, record no measurements. Exit code is non-zero if no test ran or if AsmJit
  rejected any instruction (a rejected instruction is silently absent from the JIT'd code).
  This is what CI runs; a SIGILL from a bad encoding leaves the failing test name as the last
  line of output.
- `--filter <substr>`: only run tests whose name contains the substring (use it to re-run a single
  test that crashed under `--smoke`).
- `--cpu auto|any|<n>` (Linux/Windows): which cores the main thread may use. `auto` (default)
  surveys every CPU with a pinned dependent-ADD chain, groups CPUs by L2 cluster, prints the table
  and pins to the cluster with the fastest measured clock; `<n>` pins to the cluster holding cpu
  n; `any` floats. See "Heterogeneous cores" below for why the default is measured rather than
  "cpu 0".

```bash
./arm64bench_selftest [--verbose]   # timer / PMU / calibration / harness sanity checks; sudo for PMU
```

## Architecture

| File | Purpose |
|------|---------|
| `src/main.cpp` | CLI entry point |
| `src/timer.h/.cpp` | Platform-agnostic monotonic timer |
| `src/harness.h/.cpp` | Benchmark runner and statistical analysis |
| `src/cycle_counter.h/.cpp` | PMU cycle counter abstraction (macOS kpc, Windows PMCCNTR_EL0) |
| `src/jit_buffer.h/.cpp` | JIT memory pool with W^X handling |
| `src/cpu_features.h/.cpp` | Runtime feature detection (`cpu_has(CpuFeature::X)`): OS report (sysctl / hwcap / Windows registry ID registers) confirmed by a one-instruction probe under an illegal-instruction trap |
| `src/gen_common.h/.cpp` | Shared generator scaffolding: `build_loop`, `chain_sweep`, `run_one`, `params_for`, `section`, page/pointer-ring helpers |
| `src/calibrate.cpp` | CPU frequency estimation |
| `src/gen_integer.h/.cpp` | Integer ALU latency/throughput tests |
| `src/gen_memory.h/.cpp` | Cache/memory hierarchy tests |
| `src/gen_branch.h/.cpp` | Branch prediction tests |
| `src/gen_fp_simd.h/.cpp` | FP, NEON SIMD, cross-domain, DotProd/FP16, POPCNT-idiom tests |
| `src/gen_crypto.h/.cpp` | AES/SHA-256/PMULL/CRC32, SHA3 (EOR3/BCAX/RAX1/XAR), SHA512 tests (run by `--simd`) |
| `src/gen_i8mm.h/.cpp` | FEAT_I8MM USDOT/SMMLA/UMMLA/USMMLA tests (run by `--simd`) |
| `src/gen_bf16.h/.cpp` | FEAT_BF16 BFDOT/BFMMLA/BFMLALB/BFMLALT tests (run by `--simd`) |
| `src/gen_lse.h/.cpp` | LSE atomics latency/throughput tests |
| `src/gen_pitfalls.h/.cpp` | Micro-architectural pathology tests (barriers, LRCPC, store forwarding) |
| `src/gen_ooo.h/.cpp` | Out-of-order window sizing: ROB, int/FP register files, load/store queues (two-miss probe) |
| `src/gen_sve.h/.cpp` | SVE/SVE2 tests: native with FEAT_SVE, else streaming mode via SME (Apple M4/M5) |
| `src/gen_mlp.h/.cpp` | Memory-level parallelism: K interleaved pointer chases per cache level (outstanding-miss capacity) |
| `src/gen_frontend.h/.cpp` | Decode width (NOP), MOV elimination, zero idioms, macro-op fusion pairs, branch throughput, ISB |
| `src/gen_icache.h/.cpp` | I-cache size sweep (straight-line NOP bodies), BTB chain (dense), iTLB chain (one branch per 16 KB page) |
| `src/gen_prefetch.h/.cpp` | Hardware prefetcher stride streams (asc/desc, constant footprint) and PRFM lookahead on a random chase |
| `src/gen_c2c.h/.cpp` | Core-to-core: 1-line and 2-line cache-line round trips, contended LDADDAL, against a partner thread running a JIT'd responder |
| `src/affinity.h/.cpp` | Thread placement: pin to a CPU or set (Linux/Windows), QoS cluster hint (macOS); CPU topology query (L2 clusters, efficiency class, max clock, MIDR) |
| `src/cpu_select.h/.cpp` | `--cpu`: per-CPU clock survey (pinned ADD chain), cluster choice, main-thread pinning and the topology table in the run header |
| `tests/selftest.cpp` | Self-test of the measurement machinery (timer, PMU, calibration, harness accounting) |
| `.github/workflows/ci.yml` | GitHub Actions: build + selftest + smoke on macOS/Linux/Windows arm64 runners |

## Code Conventions

- **Namespace:** `arm64bench` (generators in `arm64bench::gen`)
- **Naming:** `PascalCase` for types, `snake_case` for functions/variables, `g_` prefix for globals
- **Headers:** `#pragma once`
- **C++20** (AsmJit requires it and propagates `cxx_std_20`; designated initializers,
  `std::span`, and `<bit>` are fair game)
- **No exceptions, no RTTI** (`-fno-exceptions -fno-rtti`)
- **Feature gating:** `cpu_has(CpuFeature::X)` from `cpu_features.h`, never `__ARM_FEATURE_*`
- Section separators: `// ── Description ──────────────────────`
- Platform guards: `#if defined(_WIN32)`, `#ifdef __APPLE__`, `#ifdef __linux__`

### JIT and compile-time CPU feature macros

All test code is JIT-emitted via AsmJit. The host compiler only sees C++ method calls like `a.usdot(...)` — it never emits the target instruction itself. Therefore **compile-time feature macros (`__ARM_FEATURE_CRYPTO`, `__ARM_FEATURE_I8MM`, etc.) are never needed** to guard JIT test code. Use only runtime feature detection:

- **macOS**: `sysctlbyname("hw.optional.arm.FEAT_XXX", ...)` — comprehensive, reliable
- **Windows**: the ID_AA64ISAR0/ISAR1/PFR0_EL1 values the kernel mirrors into the registry
  (`HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, values `CP 4030` / `CP 4031` /
  `CP 4020`, REG_QWORD), decoded field by field; `IsProcessorFeaturePresent(PF_ARM_*)` only
  for SVE/SVE2, where OS support for the register state is what matters. The PF_ flags alone
  have no bit for SHA3/SHA512/FP16/FHM/BF16/LRCPC2, and an earlier "assume present" default
  for those crashed `--simd` on a Snapdragon 8cx Gen 3 (Cortex-X1C/A78C, no SHA3) with
  `0xC000001D`.
- **Linux**: `getauxval(AT_HWCAP)` / `AT_HWCAP2`
- **Every platform, on top of the OS report**: an instruction probe. Each feature except
  SVE/SVE2/SME carries one representative encoding (`kFeatures[].probe`, e.g. EOR3 for SHA3)
  that is JIT'd as `insn; ret` and executed once under an illegal-instruction trap — SEH
  `__try` on Windows, a SIGILL handler plus `sigsetjmp` elsewhere. A feature the OS reports
  present whose probe traps is treated as absent with a warning on stderr; a feature the OS
  has no report for is decided by the probe. All probes run on the first `cpu_has()` call
  (one handler install, one pass; call it from the main thread before spawning threads).
  `cpu_os_reports()` / `cpu_probe()` expose the two inputs; the selftest cross-checks them
  and FAILs on "reported present but trapped", which is the case that would otherwise kill a
  benchmark run mid-test.

All of this lives behind `cpu_has(CpuFeature::X)` in `cpu_features.h`; add a row to the table
in `cpu_features.cpp` for a new feature (sysctl name, hwcap bit, ID register field, probe
encoding — get the encoding from `clang -c -x assembler` and `objdump -d`, not by hand) rather
than writing another `#ifdef` ladder. The Linux CI leg once silently built 425 of 434 tests
because three sections were gated on `__ARM_FEATURE_*` macros the default `-march` did not
define.

Trap-and-recover is deliberately limited to the probe. Wrapping a whole JIT'd test in SEH or a
SIGILL handler would not be sound: the test functions save x19–x22/x30 in a hand-built frame
with no unwind info, so the Windows unwinder cannot get past them, and on POSIX a `siglongjmp`
out of one leaves the callee-saved registers as the JIT'd body left them. A SIGILL inside a
test means the detection table is wrong; fix the table.

## Commit Discipline

Each commit should be one logical group of changes — typically one new test
section, or one focused refactor. The constraint: **every commit must
independently compile and run.** No commit should leave the tree in a
half-wired state (e.g., a function defined but not called, or a header
declaration without an implementation). When adding several related test
sections, split them into a series of small commits — one per section —
so any single one can be reverted cleanly without unwinding the rest.
Build (`ninja`) and run the affected `--integer` / `--simd` / `--pitfalls`
flag between each commit. `ctest` (selftest + smoke) must stay green.

## Continuous Integration

`.github/workflows/ci.yml` runs one job per OS on GitHub's arm64 hosted runners:

| Runner | Hardware | Toolchain |
|---|---|---|
| `macos-15` | Apple Silicon VM (no SME exposed, so `--sve` skips) | Xcode clang, preset `macos-release` |
| `ubuntu-24.04-arm` | Azure Cobalt 100 (Neoverse N2, native SVE2 at VL 128) | apt clang, preset `linux-clang-release` |
| `windows-11-arm` | Azure Cobalt 100 (Neoverse N2), PMCCNTR_EL0 readable | clang-cl via vcvars, preset `win-clangcl-release` |

Each job builds, runs `arm64bench_selftest`, then `arm64bench --all --smoke`. A separate
`workflow_dispatch`-only job runs the full measured suite and uploads the CSV.

**What CI verifies:** every generator compiles on all three toolchains, and every JIT-emitted
instruction sequence executes on the host core (feature-detection branches included — Neoverse N2
has I8MM, BF16, LRCPC2, DotProd, so paths Apple Silicon never takes get exercised there).

**What CI cannot verify:** numbers. The macOS and Linux runners are VMs with the PMU hidden (kpc
fails even under `sudo`; `perf_event_open` is refused), so they fall to Tier 2 ratio
normalization on a shared host. The Windows runner does expose PMCCNTR_EL0 (the selftest's
PMU-implied clock matched calibration at 3.39 GHz), so that leg is Tier 1, but it is still a
shared VM. Never gate on a measured value. The selftest's PMU section reports SKIP where the
counter is hidden, and CI runs the selftest with `ARM64BENCH_SELFTEST_LENIENT=1`, which turns
measurement-quality checks (ADD ≈ 1 clk, sleep upper bounds, calibration range, ...) into WARN
lines while correctness checks still fail the job. The macOS runner has measured five 10 ms
sleeps at 46–50 ms; under that load the 5 ms reference probes are preempted more than the 2 ms
test and a chained ADD reads 0.67 clk~.

**Running the jobs locally** (`act` is installed via Homebrew, Docker Desktop provides arm64
containers):

```bash
# macOS job, directly on this machine (no container):
act push -W .github/workflows/ci.yml -j build-test --matrix os:macos-15 -P macos-15=-self-hosted

# Linux job, in an arm64 Ubuntu container:
act push -W .github/workflows/ci.yml -j build-test --matrix os:ubuntu-24.04-arm \
    -P ubuntu-24.04-arm=catthehacker/ubuntu:act-24.04 --container-architecture linux/arm64
```

The act Ubuntu image lacks cmake, which is why the Linux apt step installs it (a no-op on GitHub).
The Windows job cannot run under act; test it live on GitHub, or register a self-hosted runner in a
Windows 11 ARM64 VM (Parallels/UTM) and point `runs-on` at it temporarily.

## JIT Loop Structure

All test generators build on `gen_common.h`. Do not hand-roll a prologue/epilogue in a new
section; use the shared pieces:

- `build_loop(loops, unroll, setup, body[, scratch_bytes])` emits a fixed 48-byte frame saving
  x19–x22 and x30, `mov x19, #loops`, `setup(a)` once, a 64-byte-aligned loop top, `body(a, u)`
  for `u` in `[0, unroll)`, then `SUB x19, x19, #1` + `CBNZ` (no flag writes) and the epilogue. Bodies over 1 MB (the I-cache sweeps)
  get `CBZ done; B top` instead, since CBNZ only reaches ±1 MB.
  x20–x22 are free for the generator's constants/base addresses; x30 is saved so bodies may BL.
  `scratch_bytes > 0` reserves a 16-byte-aligned scratch area at `sp` (do `mov x9, sp` in setup).
- `chain_sweep(base, loops, unroll, "INSN tput", {2, 3, 4, 6, 8}, setup(a, nc), body(a, nc, u))`
  runs a one-instruction body over `nc` independent chains for each `nc`, rounding the unroll with
  `chain_unroll()` (a multiple of `nc`, never zero) and naming results `"<prefix> (N chains, Mx unroll)"`.
  Bodies that emit more than one instruction per `u`, or fixed heterogeneous patterns, stay on
  `build_loop`.
- `run_one(name, fn, params_for(base, loops, insns_per_loop[, bytes_per_insn]))` benchmarks,
  releases the JIT function, and returns the `BenchmarkResult` (zeroed on a compile failure,
  a filtered-out test, or smoke mode — consumers treat zero as "did not run").
- `section("Title")` prints the fixed-width header; `skip_feature(CpuFeature::X, "what")`
  prints the standard skip line.
- Registers: `xr(i)`/`wr(i)` for x0–x15, `vr(i)` for v0–v7 then v16–v24 (v8–v15 are callee-saved
  and never used). `gen_fp_simd.cpp` keeps its own S/D scalar tables with the same mapping.
- Memory: `alloc_pages`/`free_pages`/`commit_pages`, and `build_pointer_ring(buf, size, stride[,
  offset])` for random cyclic chains (fixed seed; links written with `memcpy` so misaligned nodes
  are not UB).
- Every nominal loop count must go through `scale_loops()` (or derive from `base.loops`, which
  main() already scales) so `--smoke` stays fast.
- Unroll factor varies: higher for fast instructions (ADD), lower for slow (SDIV).
- The only hand-rolled loop left is `build_rsb_chain` in `gen_branch.cpp`, which emits its
  callee functions after the outer function's RET.

## Measurement Strategy

1. Warm-up calls (default 2) to prime I-cache and prefetchers
2. Brief sleep after warm-up to stabilize CPU frequency
3. Elevate thread priority (`PriorityGuard`; on macOS uses `QOS_CLASS_USER_INTERACTIVE` to prefer P-cores)
4. Per-sample mini warm-up (1 call) immediately before each timed sample, to re-prime L1 I/D-cache after any thread migration during the inter-sample sleep
5. Tick-aligned sampling (`wait_for_tick()`) before each measurement
6. 20ms inter-sample sleep for scheduler stability
7. Discard slowest sample(s) to remove outlier preemptions
8. Report min/median ns/insn, CV%, cycles/insn (direct from PMU if available, else ratio-normalized or derived from calibrated frequency)

Console output cycle source indicators: `clk ` = PMU hardware, `clk~` = Tier 2 ratio, `clk*` = calibrated frequency, `clk?` = unknown.

## Measurement Reliability: P-state and Clock Frequency

### The core problem

CPI measurements derived from wall-clock time × calibrated CPU frequency are unreliable when the
CPU changes P-state (clock speed) during measurement. This happens in two distinct ways:

- **External throttling**: OS power manager or thermal governor changes frequency between samples.
  The 20ms inter-sample sleep is an opportunity for this. Result: some samples are measured at the
  wrong frequency, which the "discard slowest" strategy partially mitigates.

- **Instruction-induced throttling**: Certain instruction classes (e.g. Intel AVX-512 on Skylake,
  potentially wide SVE2 on some ARM µarchs) cause the CPU to throttle *while they are executing*,
  then recover quickly afterward. A reference measurement taken before/after appears unaffected
  even though the test itself ran at reduced frequency. This is silently wrong data.

### Three-tier solution

**Tier 1 — Hardware PMU cycle counters (best; P-state immune)**

Read actual CPU cycle counts from hardware counters surrounding the test. A cycle is a cycle
regardless of clock frequency, so P-state changes of any kind are irrelevant.

- **macOS**: `kpc_get_thread_counters()` via the private-but-stable `kpc` framework.
  Symbols are in `kperf.framework` on macOS ≥ 15 (Sequoia/Tahoe); in
  `libsystem_kernel.dylib` on earlier versions. The implementation tries both.
  **macOS ≥ 15 requires root for all kpc calls** (EPERM without it); earlier
  versions allowed fixed-counter reads from userspace. Run `sudo ./arm64bench`
  to enable PMU cycle counting on macOS 15+.
- **Linux**: `perf_event_open(PERF_COUNT_HW_CPU_CYCLES)` for the calling thread, user mode only,
  read with `read(2)`. VMs without a virtualized PMU and hosts with `perf_event_paranoid` above
  2 refuse the open and the harness falls back to Tier 2 (the CI runner does).
- **Windows ARM64**: `PMCCNTR_EL0` read via `__builtin_arm_rsr64` inside a SEH `__try` block.
  Requires thread affinity pinning (the counter is per-CPU; migrations between reads produce
  garbage). `QueryProcessorCycleTime` returns 100ns units, not cycles. `__rdtsc()` maps to
  the fixed-frequency generic timer (`CNTVCT_EL0`), not CPU cycles.

The `cycle_counter_available()` query in `cycle_counter.h` lets the harness select the best
available method at runtime.

**Tier 2 — Ratio normalization + reference sandwich (portable fallback)**

Normalize every test result against a 1-cycle reference instruction (ADD reg, reg, reg) measured
immediately before and after each timed sample. Clock speed cancels in the ratio:

    ratio = min(test_ns/insn over samples) / min(ref_ns/insn over all probes)

If `|ref_after - ref_before| / ref_before > threshold`, the measurement is flagged as potentially
affected by a P-state change *between* the reference probes (external throttle). The sample is
re-taken up to a retry limit.

**Limitation**: does not catch instruction-induced throttling (the test instructions change the
clock speed; both reference probes see the un-throttled rate). For instruction classes known to
risk this (SVE2 wide ops), results should be labelled as potentially reflecting throttled execution.

**Why min/min and not min of paired ratios**: on an oversubscribed host the shorter of (test,
reference) is more likely to get a preemption-free run, so the minimum of per-sample ratios is
biased toward whichever side is shorter (the macOS CI runner produced 0.67 clk~ for a chained ADD
against 5 ms reference probes). Each minimum approximates the uncontended time on its own, and on
a quiet machine the two estimators agree to three digits.

**Tier 3 — Wall-clock × calibrated frequency (current baseline)**

`min_ns_per_insn * 1e-9 * g_cpu_freq_hz`. Used when neither Tier 1 nor Tier 2 is available.
Reasonable for stable systems; unreliable under thermal pressure or for instruction-induced
throttling scenarios.

### Heterogeneous cores: pin before measuring (Linux/Windows)

The first Snapdragon result sets mixed core types test by test. `PriorityGuard` only pins to
whatever CPU the thread is on when a test starts, so the X2 Elite run alternated between 5 GHz
Prime and 3.6 GHz Performance cores (0.200 vs 0.277 ns per clk in the ns column; the Performance
core has 4 ALUs and 1 multiplier, the Prime 6 and 2) and the 8cx Gen 3 between Cortex-X1C and
A78C. CPU numbering does not help: the X2 Elite enumerates its six Performance cores as cpus 0–5
and the twelve Prime cores as 6–17, so "pin to cpu 0" picks the small cluster. `cpu_select.cpp`
therefore measures: every allowed CPU runs a pinned dependent-ADD chain for a few ms (ADD is one
cycle on every core, so instructions per ns is the clock), CPUs are grouped by L2 cluster (from
`GetLogicalProcessorInformationEx` / sysfs), and the main thread is pinned to the cluster with the
highest median clock; it may float within that cluster. The core-to-core matrix is measured from
the first CPU of that set. When reading an old result file, the ns/clk ratio per row tells you
which core type ran it.

**Windows 4 KB pages.** macOS uses 16 KB pages, Windows and most Linux distributions 4 KB. On the
X2 Elite an L2 TLB miss costs 70–90 ns (a DRAM access): the 4 KB-stride TLB chase goes from 30 clk
at 2048 pages to 96 at 4096 and 260 at 8192. Any test spanning more than ~8 MB is partly TLB-bound
there, which is why the 16 MB MLP single chain read 57 ns while the 16 MB load-latency sweep read
9 ns. DRAM-range numbers are not like-for-like across page sizes.

### Thread migration on macOS

macOS aggressively migrates threads between cores for thermal leveling. Migration during the
20ms inter-sample sleep leaves the L1 I/D-cache cold for the next sample. Mitigations:

- **`QOS_CLASS_USER_INTERACTIVE`** keeps the thread on P-cores (avoids E-core migration).
  P-cores on Apple Silicon share L2, so P→P migration only costs L1.
- **Per-sample mini warm-up** (step 4 above): one untimed call to the test function immediately
  before each timed measurement re-primes L1 after any migration. This is why warm-up must
  happen *per sample*, not only at the start of the session.

### Windows ARM64 recommendations

For the most stable measurements on Snapdragon/Windows:
- Set Power Plan to "Ultimate Performance" (`powercfg -duplicatescheme e9a42b02-...`)
- Pin thread affinity before reading PMCCNTR_EL0 (already done in `PriorityGuard`).
- Use Tier 2 ratio normalization when PMU is unavailable; treat absolute CPI numbers as approximate.

## Test Design Notes

Key micro-architectural insights that affect benchmark design:

### Load value prediction (Apple M5+)

Apple M5 (and likely M4) has an aggressive **load value predictor** that learns when a load always
returns the same value. A self-referential pointer chain `[x0] = x0` (i.e. `*x0 == x0`) triggers
this: the hardware "executes" the loads in ~0.3 clk instead of the true L1 latency (~3 clk).

**Fix**: Use a shuffled N-node ring (64 nodes, Fisher-Yates shuffle, all distinct addresses). The
ring is built on the stack and fits in L1. See `run_barrier_tests()` in `gen_pitfalls.cpp`.

**Important**: Ordered loads (LDAR, LDAPR, LDAPUR) are immune to value prediction because their
barrier semantics require actual memory completion before the result is consumable. Self-referential
chains are therefore safe for those instructions and correctly show true L1 latency.

### LDAPR/LDAPUR: what the test actually measures

In a pointer-chase chain with **no concurrent stores**, LDAPR/LDAPUR should show the same latency
as LDR and LDAR. This is **correct behavior**, not a bug — there is no store buffer to drain, so
the one-way barrier is trivially satisfied. The test result `LDAPR ≈ LDAR ≈ LDR` therefore
confirms correct implementation.

The meaningful test for LRCPC is **store-to-load forwarding**: does the CPU correctly forward
through an LDAPR/LDAPUR from a pending store? On Apple M-series, all store→ordered-load forwarding
variants (STR/STLR/STLUR → LDAPR/LDAPUR) show ~4.9 clk, identical to STR→LDR baseline, confirming
that acquire semantics do not impede store forwarding.

L1 load latency varies by chip variant due to different cache sizes. Apple M5 (base) shows ~3 clk
on the 128-bit pointer chain; M5 Pro (larger L1) shows ~5 cycles. This reflects the fundamental
latency/capacity tradeoff in SRAM design — the same pattern seen in Intel's Skylake→Ice Lake
transition.

### FEAT_I8MM: matrix multiply vs dot product on Apple M5

USDOT (unsigned×signed dot product) is the ARM equivalent of Intel's AVX-VNNI `VPDPBUSD`. It is
the canonical instruction for INT8 quantized neural network inference where activations are unsigned
and weights are signed (asymmetric quantization). SDOT/UDOT require equal signedness on both
operands, often forcing a zero-point bias correction; USDOT eliminates this overhead.

SMMLA/UMMLA/USMMLA operate on a 2×8 × 8×2 matrix layout producing a 2×2 int32 result — 32 MAC
ops per instruction (vs 16 for SDOT). However, on Apple M5, benchmarking shows:
- SMMLA latency = 6 clk (2× SDOT)
- SMMLA throughput = 1/cycle (vs 2 SDOT/cycle)
- **Net MAC throughput = identical to SDOT** (32 MACs/cycle either way)

This indicates M5 implements SMMLA as two sequential SDOT micro-ops internally. There is no
micro-architectural benefit to using SMMLA over SDOT on Apple M5. Whether Snapdragon Oryon has
dedicated matrix-multiply hardware (and therefore higher SMMLA MAC throughput) is an open question.

### Fusion tests: what throughput can and cannot show (gen_frontend.cpp)

A fused pair is one micro-op, so pairs/clk should match the cheaper single's rate. That only
discriminates when the two halves would otherwise compete for the same resource. On M5 the
branch unit and the ALUs are separate ports, so unfused CMP+B.NE can already reach the branch
rate; the measured 0.41 clk/pair versus 0.35 for B.NE alone is consistent with either. ADRP+ADD
at exactly the ADRP-alone rate, well under the sum, is the clearer case.

The width-bound test (pair + 8 or 18 NOPs per group, so the loop runs at the front-end's 10
slots/clk) asks a different question: does a fused pair save a front-end slot? On M5 the answer
is no for every pair, including AESE+AESMC, which the crypto latency test shows is fused: every
10-instruction group costs exactly 1.0 clk and every 20-instruction group 2.0. The 10/clk limit
is instruction fetch/decode, upstream of fusion; fused pairs are one micro-op downstream but
still two instructions to the front end. A core that fuses in decode and renames narrower than
it decodes would show 0.9 there.

### Pointer-chase stride and set conflicts

`gen_memory.cpp`'s latency sweep chases one node per cache line (64 B stride), so a buffer of B
bytes occupies B bytes of every level and the boundaries land at the real capacities (M5: 3 clk
through 128 KB, 12.9 clk at 256 KB). The other chase tests (`gen_ooo`, `gen_mlp`, `gen_prefetch`)
use a 256 B node stride. That touches only every fourth set, so a buffer of B bytes holds B/4
bytes of data in the cache; the level a buffer lands in is still decided by lines per set, so
the boundary in *buffer* bytes is unchanged (a 128 KB buffer fits an 8-way 128 KB L1 at either
stride), but anything that reasons in bytes of data (footprint, bandwidth, "how much of L2 is
this") must divide by four. `gen_mlp.cpp` picks its footprints with that in mind (64 KB =
L1-resident control, 2 MB = L2). If a test needs the full data capacity, use a 64 B stride;
the random permutation defeats the next-line prefetcher either way.

### Out-of-order window probe: lessons (gen_ooo.cpp)

- **Every timed call must traverse the whole pointer ring.** A partial walk revisits the same
  nodes every call; the per-sample warm-up then leaves them in L2 and a "DRAM miss" silently
  becomes an L2 hit (27 ns instead of 81 ns on M5). `loops = ring_nodes / misses_per_iteration`.
- **Check overlap before trusting knees:** two independent chains with no fillers must cost the
  same as one (`overlap factor ≈ 1.0`). If it is ~2–3×, the misses are not real misses.
- **Fillers must not touch the structure you are not measuring.** `LDR xN` consumes an integer
  physical register, so its knee is the int PRF; `LDR XZR` is a load-queue entry with no register.
  NOPs on M5 show no limit to 2048 — they are apparently dropped before allocation.
- **Two dependent misses per chain** double the shadow (~750 clk) so 2 × 2048 fillers stay inside it.
- **Flags and branches are structures too.** `CMP x2, x3` allocates a flag physical register and
  nothing else (knee ≈ 170–179 on M5); a never-taken `B.EQ` (flags set NE once in setup) allocates a
  branch-order-buffer entry (knee ≈ 194–203; a taken-to-next-instruction B.NE gave the same knee on M5
  but 42–51 vs 154–163 on the two Neoverse N2 CI legs, so keep it not taken). Both are smaller than the integer PRF, so a filler that sets flags or branches
  measures those, not the PRF or ROB.
- **Masked pointers** (`link ^ 0xA5A5…`, unmasked with EOR) defeat any data-dependent prefetcher;
  M5 showed no plain-vs-masked difference (DIT made none either), but the sweeps use masked rings.

### Streaming SVE via SME (gen_sve.cpp)

Apple M4/M5 have no FEAT_SVE but do have FEAT_SME, whose streaming mode executes the SVE
instruction set (minus a few instructions) on the SME unit at the streaming vector length
(512 bits on M5). `--sve` uses that when native SVE is absent: each JIT'd function does
`SMSTART SM` in setup and `SMSTOP SM` in teardown. Two traps:

- **SMSTART/SMSTOP zero every vector register, including callee-saved d8–d15.** A JIT'd function
  that enters streaming mode must save d8–d15 before SMSTART and restore them after SMSTOP, or the
  C++ caller's state is silently destroyed. The symptom here was surreal: clang kept an AsmJit
  operand signature in d8 across calls, so after the first streaming-mode call every later
  `Imm(...)` became a "none" operand and every mov-immediate was silently dropped from the JIT'd
  code — a loop with no counter and no base address. `gen_sve.cpp::emit_sm_enter/leave` do the
  save/restore; the JIT pool now installs an AsmJit error handler so a dropped instruction is at
  least printed.
- **NEON is mostly illegal in streaming mode.** Nothing between SMSTART and SMSTOP may use v-register
  instructions; the harness reference function runs outside the JIT'd function and is fine.

M5 streaming-mode results (VL 512): ADD z.s 3.1 clk / 1 per clk; FADD/FMUL/FMLA/SDOT z.s 8.3 clk
latency, one per ~4.2 clk regardless of chain count (≈4 f32 FMLA lanes/clk, about a quarter of the
NEON pipes); LD1W/LDR z ≈1 clk per 64 B (265 GB/s); stores over a ≥16 KB window: ST1W, STR z and
STNT1W all 1.0 clk per 64 B store (PMU run; an unprivileged Tier 2 run read STR z/STNT1W at 2.8–3.4,
bimodally), STR q 3.9 clk, versus 0.5 clk for STR q outside streaming mode (see the next paragraph
for why the window matters);
WHILELT/PTRUE ≈1 clk. These numbers are core-clock units (Tier 2 ratio), not SME-unit clocks.

**Streaming-mode store serialization.** In streaming mode every SIMD&FP-register store (ST1W,
STR z, STNT1W, and plain STR q alike) stalls when it hits a line that a recent SIMD store wrote:
≈104 clk per store when every store hits one line (some runs settle at ≈29 clk instead — the
same-line case is bimodal run to run, each mode stable to 0.1 %), ≈52 clk rotating over 4 lines,
≈14 clk over 16, ≈3.4 clk over 64, and full rate only once ≈256 distinct lines (16 KB) separate
rewrites (PMU-backed run).
Scalar STR x is unaffected (0.5 clk at every window, in or out of streaming mode), and STR q
outside streaming mode is 0.5 clk at every window, so it is the mode's SIMD store path, not
the addresses. The store-rotation sweep in `gen_sve.cpp` shows the whole curve. Any streaming
SVE store test therefore needs a wide address rotation, and streaming-mode code that
accumulates in memory (small in-place buffers) pays this in production too. Even at full rate a
streaming SIMD store costs 2–5× a NEON-mode STR q.

ST1W/STNT1W take an immediate of only −8..7 vectors (STR z reaches ±256). A first version of the
sweep used one base register and read 0.3 clk per ST1W over 64 KB: three quarters of the stores
had been rejected by AsmJit and the loop ran a quarter of the work. `build_store_loop` spreads
the iteration base over one register per 8 vectors, and a measured run now exits non-zero, with
a warning, if AsmJit rejected anything.

### Core-to-core tests (gen_c2c.cpp)

A partner thread runs a JIT'd responder (spin on a line, reply, check a stop line only while
idle) while the main thread runs the timed function through the ordinary harness. Three things
keep that honest:

- **Protocol counters are never reset.** The timed function loads the current counter in its
  setup and continues from it, so warm-up calls, reference calls and timed calls all agree with
  the responder whatever ran before; a reset between calls would race the responder's last reply.
- **Placement.** Linux and Windows pin both threads (`affinity.h`); the main thread sits on the
  first allowed CPU and the partner sweeps the rest. macOS has no affinity API for user threads
  (`thread_affinity_policy` is a hint Apple Silicon ignores), so the partner is placed by QoS
  class instead: user-interactive lands on the performance cluster, background on the efficiency
  cluster. That gives P↔P and P↔E, with the exact cores the scheduler's choice.
- **A single available CPU skips the section.** A spinning partner sharing the main thread's
  core turns every hop into a timeslice.

The contended-LDADDAL line is inherently noisy (CoV 20–40 % on M5): line ownership alternates in
bursts, so the main thread's share of the increments varies sample to sample. Read it against the
no-partner reference line, not to three digits.

### Prefetcher stride sweep: not every stride is a stride (gen_prefetch.cpp)

On M5 the sweep separates two mechanisms. Strides up to 256 B (including 192 B, three lines) are
followed at 9–18 ns/load. Power-of-two strides from 1 KB to 32 KB are followed too, across page
boundaries, at 13–27 ns. The non-power-of-two strides between them are not: 384 B, 768 B and
1536 B run at 46–76 ns, near the 88 ns unassisted chase, in both directions, and 512 B lands in
the middle at 28 ns. A classic stride detector has no reason to prefer 1024 over 768; a spatial
prefetcher that learns which lines of a fixed-size region were touched and replays that pattern
on the next region does, because only a stride that divides the region size produces the same
pattern in every region. Sweeps that only try powers of two would call this prefetcher perfect.

### AsmJit API notes

- `Gp` not `GpX` for general-purpose register arguments in helper functions
- `a.embed(&word, 4)` to hand-encode instructions not exposed in AsmJit's C++ API
  (used for LDAPR, LDAPUR, STLUR in `gen_pitfalls.cpp`)
- `a.ldr(xzr, ptr(xN))` encodes (LDR to XZR: load and discard, no register written)
- `a.fmov(sN, 0.0)` is NOT encodable (FMOV immediate has no zero); use `movi(vs4(N), Imm(0))`.
  AsmJit rejects it, and before the JIT pool had an error handler the instruction vanished silently.
- NEON XAR was mis-encoded in upstream asmjit since 2022 (RAX1's opcode bits); fixed in the fork.
- `stlxr wS, xT, [xN]` (64-bit data) was rejected by upstream asmjit (classified under an encoder
  that requires equal-width status and data registers); fixed in the fork. `--smoke` now fails
  if AsmJit rejected any instruction, so a dropped instruction can no longer pass CI.
  SHA3: `eor3/bcax(vd, vn, vm, va)` all `.b16()`, `rax1(vd.d2(), vn.d2(), vm.d2())`,
  `xar(vd.d2(), vn.d2(), vm.d2(), Imm(rot))`; SHA512: `sha512h(vd.q(), vn.q(), vm.d2())`,
  `sha512su0(vd.d2(), vn.d2())`; BF16: `bfdot/bfmmla/bfmlalb(vd.s4(), vn.h8(), vm.h8())`,
  bf16(0.5) = `movi(v.h8(), Imm(0x3F), Imm(8))`; `fjcvtzs(w0, d0)`.
- SVE (fork): `z0.s()`/`.b()/.h()/.d()` element views; `p0.m()`/`p0.z()` governing predicates,
  `p0.s()` for PTRUE/WHILELT; `fmla(z0.s(), p0.m(), z1.s(), z2.s())`; `dup(z0.s(), w9)` broadcast
  from GPR, `dup(z0.s(), z0.s(0))` lane broadcast; `faddv(s0, p0, z0.s())`; `ld1w(z0.s(), p0.z(),
  ptr_vl(x20, k))` / `st1w(z0.s(), p0, ptr_vl(...))` / `ldr(z0, ptr_vl(...))`; `whilelt(p1.s(), w2, w3)`;
  `rdvl(x0, 1)`; `smstart_sm()`/`smstop_sm()`. Every encoding checked so far matched the ARM ARM.
- `MSR DIT, #imm` = `0xD503405F | (imm << 8)` if ever needed (FEAT_DIT; no effect seen on M5)
- AESE/AESMC: `.b16()` element type
- PMULL poly64: `.q()` result, `.d()` inputs
- SHA256H: `.q()` first two args, `.s4()` third
- `movi(vec, Imm(0x3C), Imm(8))` for fp16(1.0) initialization (MOVI with shift)
- SDOT/USDOT/SMMLA/UMMLA/USMMLA: `.s4()` accumulator, `.b16()` byte inputs

## Completed Test Coverage

| Category | File | Key Results (Apple M5) |
|---|---|---|
| **Integer ALU** | `gen_integer.cpp` | ADD=1 clk, MUL=3 clk, SDIV/UDIV=7 clk (UDIV 2 per clk with 4 chains), EXTR=2 clk, UMULH=3 clk |
| **CSEL/CSINV/CSNEG** | `gen_integer.cpp §9` | ADDS+CSEL chain 0.73–0.76 clk/insn (PMU and Tier 2 agree), i.e. CSEL with Xd=Xn on the true arm adds ≈0.5 clk on average, not a full cycle; CSINV/CSNEG add the full 1 clk. An earlier run read 0.53/insn and was recorded as "latency 0" |
| **Branch prediction** | `gen_branch.cpp` | Various predictor stress tests |
| **Cache hierarchy** | `gen_memory.cpp` | L1/L2/L3/DRAM latency and bandwidth sweeps |
| **TLB hierarchy** | `gen_memory.cpp` | L1 DTLB ~32 entries, L2 TLB ~256–512 entries; L1 hit=3 clk, L2 hit=11 clk |
| **LDP/STP copy** | `gen_memory.cpp` | L1=133 GB/s, L2=58 GB/s, L3/SLC=40–48 GB/s |
| **LDNP bandwidth** | `gen_memory.cpp` | Identical to LDP (Apple Silicon ignores non-temporal hint) |
| **LSE atomics** | `gen_lse.cpp` | LDADDAL=7 clk, SWPAL=2.5 clk, LDAXR+STLXR=16 clk (earlier "11 clk" was LDAXR alone: asmjit had silently dropped the STLXR) |
| **Scalar FP** | `gen_fp_simd.cpp §1–2` | FMUL f32/f64=3 clk, FDIV f32=7 clk, FSQRT f32=9 clk |
| **NEON FP** | `gen_fp_simd.cpp §3–4` | FMLA v4f32=3 clk; 4 pipes, but only visible with 16 chains (0.26 clk/insn; 6 chains still read 0.57, latency-bound). Sweeps go to 16 chains since 2026-09-09 |
| **Cross-domain** | `gen_fp_simd.cpp §7` | FMOV GPR↔FP=5 clk, SCVTF/FCVTZS=6 clk |
| **Crypto (AES/SHA/CRC)** | `gen_fp_simd.cpp §8` | AESE+AESMC fused=2.1 clk/pair, PMULL=3 clk, SHA256H=4 clk, CRC32=3 clk |
| **SDOT/UDOT/SMLAL/FP16** | `gen_fp_simd.cpp §9` | SDOT/UDOT=3 clk, FMLA v8f16=3 clk (uniform FMA latency all precisions) |
| **FEAT_I8MM** | `gen_fp_simd.cpp §10` | USDOT=3 clk, SMMLA/UMMLA/USMMLA=6 clk, same MAC throughput as SDOT |
| **Bitfield (BFI/BFXIL/UBFX/SBFX)** | `gen_integer.cpp §10` | All ~1 clk latency; BFI throughput stays at 1 clk regardless of chains (single BFI unit) |
| **Misc bit-ops (CLS/BIC/ORN/EON/CCMP)** | `gen_integer.cpp §11` | CLS≈CLZ at 1 clk; BIC saturates ~4 chains; CCMP flag chain ~0.4 clk avg |
| **POPCNT idiom** | `gen_fp_simd.cpp §11` | NEON CNT v16b=2 clk; full FMOV+CNT+ADDV+FMOV scalar-POPCNT idiom ≈14.8 clk per emulated POPCNT |
| **Memory barriers** | `gen_pitfalls.cpp §5` | DMB=1.5 clk standalone; in load chain: 0 added (completes within LDR latency) |
| **LRCPC (LDAPR/LDAPUR)** | `gen_pitfalls.cpp §6` | LDAPR≈LDAR≈LDR=3 clk; store forwarding unchanged (~4.9 clk all variants) |
| **BFI dest-dep stress** | `gen_pitfalls.cpp §7` | All three Mihocka variants (independent / overlapping rotation / full-width) report ~1 clk on M5 — no dep-breaking shortcut |
| **OOO window** | `gen_ooo.cpp` | Two-miss probe: int PRF ≈ 386–418, FP PRF ≈ 834–898, load queue ≈ 482–515, store queue ≈ 138–146, flag (NZCV) PRF ≈ 170–179 (CMP fill), branch order buffer ≈ 194–203 (not-taken B.cond fill); NOP fill shows no limit to 2048 (NOPs are not allocated, or ROB > 2050). Sharp 1×→2× steps. ~3.5 min run |
| **SHA3 / SHA512** | `gen_crypto.cpp` | EOR3, BCAX, RAX1, XAR all 2 clk, saturate at 6 chains ≈0.33 clk (3 units); SHA512H/H2/SU0/SU1 all 2 clk |
| **FEAT_BF16** | `gen_bf16.cpp` | BFDOT 3 clk, 1/clk (half the SDOT rate); BFMMLA 4.9 clk, 1 per 2 clk — same 8 MAC/clk either way, no matrix-form advantage (as with SMMLA); BFMLALB/T 4 clk, ~1.5/clk |
| **JSCVT** | `gen_fp_simd.cpp §7` | SCVTF/FJCVTZS round trip 6.0 clk = same as SCVTF/FCVTZS (5.9); JavaScript ToInt32 semantics are free |
| **Memory-level parallelism** | `gen_mlp.cpp` | Effective MLP (1-chain latency / saturated per-load time): 2 MB ≈ 6.7, 16 MB ≈ 11, DRAM ≈ 18 misses in flight (13 GB/s random lines, floor leaves at 22–24 chains); L1 dependent loads issue at 1/clk |
| **Front-end / rename** | `gen_frontend.cpp` | 10 NOPs/clk at every body size; GPR MOV eliminated only when consumed by an ALU op (pure MOV chain 0.9 clk); FMOV d,d and ORR v,v 2 clk (executed); **no zero idioms** (EOR/SUB/AND-xzr, vector EOR/SUB all stay dependent); ADRP+ADD pairs = ADRP alone; MOVZ 8.8/clk without an ALU; no pair (CMP/SUBS+B.cond, ADD+CBZ, ADRP+ADD, MOVZ+MOVK, AESE+AESMC) saves a front-end slot (10-insn groups all exactly 1.0 clk); 2 taken B/clk; ISB 34 clk |
| **I-cache / BTB / iTLB** | `gen_icache.cpp` | L1I 192 KB (10 NOP/clk to 192 KB, 3.2/clk from L2, ~2/clk at 16 MB); BTB: zero-bubble taken branches to 48–64 sites, 2–3 clk to ~384, 4.2 clk beyond; L1 iTLB ≥ 192 × 16 KB pages, L2 TLB +9 clk from 256 to ≥ 2048 pages |
| **FP width conversions** | `gen_fp_simd.cpp §8` | FCVTL/FCVTN (f16↔f32, f32↔f64, low and high halves) and scalar FCVT all 3 clk latency, 4 per clk; FCVTN2's destination merge is free |
| **Core-to-core** | `gen_c2c.cpp` | Unpinned (QoS-placed) on M5: P↔P round trip ≈ 104 ns (~52 ns one way), P↔E ≈ 320 ns; identical for LDAR/STLR, LDR/STR, 1-line and 2-line; LDADDAL 7 clk alone, ≈ 6.5–9 ns contended (CoV 20–40 %, arbitration is bursty). Pinned core matrices come from Linux/Windows |
| **Prefetcher** | `gen_prefetch.cpp` | Power-of-two strides 64 B–32 KB and 192 B followed both directions across 16 KB pages (9–27 ns/load vs 88 ns random); 384 B, 768 B, 1536 B are NOT followed (46–76 ns), 512 B half-followed (28 ns): consistent with a short-stride detector up to ~256 B plus a spatial-pattern prefetcher that only matches when the stride tiles the region; PRFM honored, scales as latency/D: 45 ns at D=2, 12.8 at D=8, 5.1 at D=32 (= the MLP floor) |
| **SVE (streaming via SME)** | `gen_sve.cpp` | VL 512: FADD/FMLA/SDOT z.s 8.3 clk, 1 per 4.2 clk; ADD z.s 3.1 clk; LD1W 265 GB/s; ST1W/STR z/STNT1W 1.0 clk per 64 B store over a ≥16 KB window but 29–104 clk when rewriting one line (SIMD stores serialize in streaming mode, STR q too; scalar STR x immune); WHILELT/PTRUE 1 clk. Native SVE numbers (Neoverse N2) come from CI |

## Planned Test Coverage

| Category | Tests | Notes |
|---|---|---|
| **OOO window, ROB itself** | ROB via a non-NOP filler | Every non-NOP filler tried allocates a smaller structure first (int/FP/flag PRF, LQ/SQ, BOB); the ROB knee needs a filler with no other footprint that Apple does not drop |
| **FEAT_LRCPC3** | LDIAPP / STILP pair instructions | Not present on any current Apple Silicon (M1–M5); available check via `hw.optional.arm.FEAT_LRCPC3` |
| **SVE2, more** | Gather/scatter, MOVPRFX fusion, BFMMLA z, predicate-heavy loops | Native on CI (N2, 128-bit); streaming on M4/M5. Wide native SVE may need PMU (Tier 1) to be trustworthy — instruction-induced throttling risk |

## Cross-platform results (Snapdragon, 2026-09-09)

Full result files live in `~/Nextcloud/arm64bench/` (human-readable console output, one per
machine). Both Snapdragon runs predate `--cpu` pinning and mix core types row by row; the Prime /
X1C rows are identified by ns per clk (0.200 / 0.334). Headline findings, Prime or X1C core:

| | 8cx Gen 3 (Cortex-X1C, 3.0 GHz) | X2 Elite (Oryon v3 Prime, 5.0 GHz) | M5 |
|---|---|---|---|
| Integer ALUs | 4 | 6 (Performance core: 4) | ~7 |
| MUL lat / units; MADD acc chain | 2 / 2; 1 clk | 3 / 2 (Perf: 1); **3 clk** | 3 / ~3; 1 clk |
| UDIV | 6 clk, not pipelined | 7 clk, 2 clk tput | 7 clk, 2 clk tput |
| FADD / FMUL / FMLA latency | 2 / 3 / 2 | 3 / 4 / 3 | 2 / 3 / 3 |
| SDOT latency, rate | 2, 2/clk | 2, 4/clk | 3, 2/clk |
| **SMMLA MACs/clk vs SDOT** | skipped (no I8MM) | **2×** (2 clk latency, same issue rate: real matrix hardware) | 1× (two SDOT µops) |
| BFMMLA | skipped | 8 clk, ~0.75/clk | 4.9 clk, 0.5/clk |
| L1D latency / size | 4 clk / 64 KB | 3 clk; the 64 B chase stays 3 clk to 512 KB (see below) | 3 clk / 128 KB |
| DRAM latency (128 MB) | 203 ns | 126 ns | 77 ns |
| Single-core DRAM read BW | 21 GB/s | 78 GB/s | 81 GB/s |
| L1 load BW | 16 B/clk (one LDP/clk) | 64 B/clk | 48 B/clk |
| LDADD relaxed, L1-hot | 13 clk | **18.3 clk, every LSE op and CAS alike; LDAXR+STLXR 16** | 7 clk |
| ROB / int PRF / FP PRF | ~430 NOPs / 130–160 / 200–220 | 640–700 / 290–380 / 260–380 | >2048 / 386–418 / 834–898 |
| LQ / SQ / flag / BOB | 120–150 / 56–64 / ~60 / ~224 | 128–136 / 112–120 / 112–120 / 128–136 (suspiciously one band) | 482–515 / 138–146 / 170–179 / 194–203 |
| RSB | ~16 | ~50 | 64 |
| Taken branches/clk | 1 | 1 | 2 |
| BLR, one site cycling N targets | predicted to 256 targets, 6 clk | predicted to 24 targets, 7 clk; 23 clk beyond | mispredicts from 2 targets (18–24 clk): no target history |
| Indirect fast table (N sites → N targets) | 48 | 32 | ~140 |
| DMB ISH / DSB ISH / ISB | 7 / 7 / 25 | 2 / 2 / 21 | 1.6 / 18 / 34 |
| Store→load forwarding | 4.1–4.4 clk; narrow→wide 10.4 | 6.0 clk for every case | 4.1 matched x64; 0.8–1.2 for narrower loads; narrow→wide 2.6 |
| Non-temporal hints | ignored | honored: STNP to 64 MB at the L1 rate (2.1 vs 4.9 clk), LDNP does not allocate in L2 | STNP partly (3.4 vs 4.7), LDNP ignored |
| Stride prefetcher | every stride to 32 KB, across pages | ≤ 512 B only | ≤ 256 B plus power-of-two to 32 KB |
| Core-to-core round trip | 155–165 ns, all pairs (one DSU) | 60 ns in-cluster, 420 ns cross-cluster (measured from a Performance core) | 107 P↔P, 380 P↔E |

Open questions from those runs: the X2 Elite 64 B-stride random chase reads 3 clk through 512 KB
while its 4 KB-stride TLB chase already pays 18 clk at 32 pages, so something replays the dense
ring (a pointer-following prefetcher would; the masked-link sweep in `gen_memory` exists to test
this); its 3.6 GHz Performance core reads dependent ADD-immediate chains at 0.73–0.87 clk while
register chains read 1.0 (immediate merging, or a PMU quirk); the 8cx's 16 B/clk L1 load
bandwidth may be LDP X-pair single-issue on X1C or a set conflict in the 32-stream pattern.

## Feature Detection Reference

Windows column: the ID register field (from the registry mirror) is the authority; the PF_ flag
is a fallback if the registry value cannot be read; the probe instruction is executed on every
platform to confirm. See "JIT and compile-time CPU feature macros" above.

| Feature | macOS sysctl | Windows ID register field | Probe |
|---|---|---|---|
| AES / PMULL / SHA256 / CRC32 | `hw.optional.arm.FEAT_AES` etc. | ISAR0.AES ≥ 1 / ≥ 2, SHA2 ≥ 1, CRC32 ≥ 1 (PF 30/31 fallback) | AESE / PMULL / SHA256SU0 / CRC32X |
| FEAT_LSE | `hw.optional.arm.FEAT_LSE` | ISAR0.Atomic ≥ 2 (PF 34) | LDADD |
| FEAT_DotProd | `hw.optional.arm.FEAT_DotProd` | ISAR0.DP ≥ 1 (PF 43) | SDOT |
| FEAT_FP16 / FHM | `FEAT_FP16` / `FEAT_FHM` | PFR0.AdvSIMD == 1 / ISAR0.FHM ≥ 1 (no PF flag) | FADD v8h / FMLAL |
| FEAT_JSCVT | `FEAT_JSCVT` | ISAR1.JSCVT ≥ 1 (PF 44) | FJCVTZS |
| FEAT_I8MM | `FEAT_I8MM` | ISAR1.I8MM ≥ 1 (the old `PF_ARM_SVE_I8MM` proxy was dropped: Oryon has I8MM without SVE) | USDOT |
| FEAT_BF16 | `FEAT_BF16` | ISAR1.BF16 ≥ 1 (no PF flag) | BFDOT |
| FEAT_SHA3 / SHA512 | `FEAT_SHA3` / `FEAT_SHA512` | ISAR0.SHA3 ≥ 1 / ISAR0.SHA2 ≥ 2 (no PF flag; absent on 8cx Gen 3) | EOR3 / SHA512SU0 |
| FEAT_LRCPC / LRCPC2 / LRCPC3 | `FEAT_LRCPC` / `FEAT_LRCPC2` / `FEAT_LRCPC3` | ISAR1.LRCPC ≥ 1 / ≥ 2 / ≥ 3 (PF 45 for LRCPC) | LDAPR / LDAPUR / LDIAPP |
| FEAT_SVE / SVE2 | `hw.optional.arm.FEAT_SVE` (absent on Apple) | `PF_ARM_SVE_INSTRUCTIONS_AVAILABLE` (46) / `PF_ARM_SVE2_…` (47) only — OS enablement | none |
| FEAT_SME | `hw.optional.arm.FEAT_SME` (M4+) | no source; Unknown with no probe → absent | none |
