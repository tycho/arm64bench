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
./arm64bench [--all | --integer | --memory | --branch | --simd | --lse | --pitfalls | --ooo | --sve | --mlp]
             [--MHz <freq>] [--samples <n>] [--warmup <n>] [--csv]
             [--smoke] [--filter <substr>]
```

Default (no flags): runs integer and memory tests.

- `--smoke`: execute every generated test function once with loop counts divided by 1000, print
  `ok` per test, record no measurements. Exit code is non-zero if no test ran or if AsmJit
  rejected any instruction (a rejected instruction is silently absent from the JIT'd code).
  This is what CI runs; a SIGILL from a bad encoding leaves the failing test name as the last
  line of output.
- `--filter <substr>`: only run tests whose name contains the substring (use it to re-run a single
  test that crashed under `--smoke`).

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
| `src/cpu_features.h/.cpp` | Runtime feature detection (`cpu_has(CpuFeature::X)`) for macOS/Linux/Windows |
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
- **Windows**: `IsProcessorFeaturePresent(PF_ARM_*)` — limited coverage; see notes per-feature
- **Linux**: `getauxval(AT_HWCAP)` / `AT_HWCAP2`

All of this lives behind `cpu_has(CpuFeature::X)` in `cpu_features.h`; add a row to the table
in `cpu_features.cpp` for a new feature rather than writing another `#ifdef` ladder. The Linux
CI leg once silently built 425 of 434 tests because three sections were gated on
`__ARM_FEATURE_*` macros the default `-march` did not define.

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
| `macos-15` | Apple Silicon VM | Xcode clang, preset `macos-release` |
| `ubuntu-24.04-arm` | Azure Cobalt 100 (Neoverse N2) | apt clang, preset `linux-clang-release` |
| `windows-11-arm` | Azure Cobalt 100 (Neoverse N2) | clang-cl via vcvars, preset `win-clangcl-release` |

Each job builds, runs `arm64bench_selftest`, then `arm64bench --all --smoke`. A separate
`workflow_dispatch`-only job runs the full measured suite and uploads the CSV.

**What CI verifies:** every generator compiles on all three toolchains, and every JIT-emitted
instruction sequence executes on the host core (feature-detection branches included — Neoverse N2
has I8MM, BF16, LRCPC2, DotProd, so paths Apple Silicon never takes get exercised there).

**What CI cannot verify:** numbers. All three runners are VMs with the PMU hidden (kpc fails even
under `sudo`, PMCCNTR_EL0 traps), so everything falls to Tier 2 ratio normalization on a shared
host. Never gate on a measured value. The selftest's PMU section reports SKIP there, and CI runs
the selftest with `ARM64BENCH_SELFTEST_LENIENT=1`, which turns measurement-quality checks (ADD ≈
1 clk, sleep upper bounds, calibration range, ...) into WARN lines while correctness checks still
fail the job. The macOS runner has measured five 10 ms sleeps at 46–50 ms; under that load the
5 ms reference probes are preempted more than the 2 ms test and a chained ADD reads 0.67 clk~.

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
  for `u` in `[0, unroll)`, then `SUB x19, x19, #1` + `CBNZ` (no flag writes) and the epilogue.
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
- **Linux**: `perf_event_open(PERF_COUNT_HW_CPU_CYCLES)` per-thread (not yet implemented).
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

### Pointer-chase stride and L1 set conflicts

Every pointer-chase test uses a 256-byte node stride. A 128 KB 8-way L1D has 256 sets of 64 B;
a 256 B stride touches only every fourth set, so the chase sees an L1 of 32 KB (8 ways × 64
sets), and the line footprint is buffer/4. That is why the latency sweep in `gen_memory.cpp`
shows the L1 "boundary" at a 256 KB buffer on a 128 KB cache. It is consistent and the L2/DRAM
levels are unaffected, but treat buffer sizes below ~1 MB as "quarter-L1" numbers, and if a
test needs the real L1 capacity use a 64 B or 128 B stride (with a random permutation the
next-line prefetcher cannot follow it anyway). `gen_mlp.cpp` picks its footprints with this in
mind (64 KB = L1-resident control, 2 MB = L2).

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
NEON pipes); LD1W/LDR z ≈1 clk per 64 B (265 GB/s); ST1W 57 clk per store (4.9 GB/s) and STR z
bimodal 28–57 clk — streaming-mode stores are pathologically slow and deserve a closer look;
WHILELT/PTRUE ≈1 clk. These numbers are core-clock units (Tier 2 ratio), not SME-unit clocks.

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
| **Integer ALU** | `gen_integer.cpp` | ADD=1 clk, MUL=3 clk, SDIV=10 clk, EXTR=2 clk, UMULH=3 clk |
| **CSEL/CSINV/CSNEG** | `gen_integer.cpp §9` | CSEL true-arm latency=0 (M5 mux bypass when Xd=Xn); CSINV/CSNEG=1 clk |
| **Branch prediction** | `gen_branch.cpp` | Various predictor stress tests |
| **Cache hierarchy** | `gen_memory.cpp` | L1/L2/L3/DRAM latency and bandwidth sweeps |
| **TLB hierarchy** | `gen_memory.cpp` | L1 DTLB ~32 entries, L2 TLB ~256–512 entries; L1 hit=3 clk, L2 hit=11 clk |
| **LDP/STP copy** | `gen_memory.cpp` | L1=133 GB/s, L2=58 GB/s, L3/SLC=40–48 GB/s |
| **LDNP bandwidth** | `gen_memory.cpp` | Identical to LDP (Apple Silicon ignores non-temporal hint) |
| **LSE atomics** | `gen_lse.cpp` | LDADDAL=7 clk, SWPAL=2.5 clk, LDAXR+STLXR=16 clk (earlier "11 clk" was LDAXR alone: asmjit had silently dropped the STLXR) |
| **Scalar FP** | `gen_fp_simd.cpp §1–2` | FMUL f32/f64=3 clk, FDIV f32=7 clk, FSQRT f32=9 clk |
| **NEON FP** | `gen_fp_simd.cpp §3–4` | FMLA v4f32=3 clk; throughput saturates at 4 chains (~4 FP units) |
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
| **OOO window** | `gen_ooo.cpp` | Two-miss probe: int PRF ≈ 386–418, FP PRF ≈ 834–898, load queue ≈ 482–515, store queue ≈ 138–146; NOP fill shows no limit to 2048 (NOPs are not allocated, or ROB > 2050). Sharp 1×→2× steps. ~2.5 min run |
| **SHA3 / SHA512** | `gen_crypto.cpp` | EOR3, BCAX, RAX1, XAR all 2 clk, saturate at 6 chains ≈0.33 clk (3 units); SHA512H/H2/SU0/SU1 all 2 clk |
| **FEAT_BF16** | `gen_bf16.cpp` | BFDOT 3 clk, 1/clk (half the SDOT rate); BFMMLA 4.9 clk, 1 per 2 clk — same 8 MAC/clk either way, no matrix-form advantage (as with SMMLA); BFMLALB/T 4 clk, ~1.5/clk |
| **JSCVT** | `gen_fp_simd.cpp §7` | SCVTF/FJCVTZS round trip 6.0 clk = same as SCVTF/FCVTZS (5.9); JavaScript ToInt32 semantics are free |
| **Memory-level parallelism** | `gen_mlp.cpp` | Effective MLP (1-chain latency / saturated per-load time): 2 MB ≈ 6.7, 16 MB ≈ 11, DRAM ≈ 18 misses in flight (13 GB/s random lines, floor leaves at 22–24 chains); L1 dependent loads issue at 1/clk |
| **SVE (streaming via SME)** | `gen_sve.cpp` | VL 512: FADD/FMLA/SDOT z.s 8.3 clk, 1 per 4.2 clk; ADD z.s 3.1 clk; LD1W 265 GB/s; ST1W 57 clk/store (!); WHILELT/PTRUE 1 clk. Native SVE numbers (Neoverse N2) come from CI |

## Planned Test Coverage

| Category | Tests | Notes |
|---|---|---|
| **Prefetcher** | Stride sweep, descending scan, PRFM effectiveness | How far ahead does the hardware prefetcher reach? |
| **OOO window, more fillers** | Branch-order buffer, flag PRF, ROB via non-NOP filler | `gen_ooo.cpp` has the machinery; needs a filler with no PRF/queue footprint that Apple does not eliminate |
| **FEAT_LRCPC3** | LDIAPP / STILP pair instructions | Not present on any current Apple Silicon (M1–M5); available check via `hw.optional.arm.FEAT_LRCPC3` |
| **SVE2, more** | Gather/scatter, MOVPRFX fusion, BFMMLA z, predicate-heavy loops, streaming-mode store pathology | Native on CI (N2, 128-bit); streaming on M4/M5. Wide native SVE may need PMU (Tier 1) to be trustworthy — instruction-induced throttling risk |
| **SDOT/SMMLA cross-platform** | Compare MAC throughput on Snapdragon X | Does Oryon have dedicated SMMLA hardware, or also micro-op fusion like M5? |
| **FCVTL/FCVTN** | FP16↔FP32 conversion throughput | Widening/narrowing pipeline characterization |

## Feature Detection Reference

| Feature | macOS sysctl | Windows |
|---|---|---|
| FEAT_I8MM | `hw.optional.arm.FEAT_I8MM` | `PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE` (proxy; no direct PF_ exists) |
| FEAT_LRCPC | `hw.optional.arm.FEAT_LRCPC` | assume true (Oryon) |
| FEAT_LRCPC2 | `hw.optional.arm.FEAT_LRCPC2` | assume true (Oryon) |
| FEAT_LRCPC3 | `hw.optional.arm.FEAT_LRCPC3` | unknown |
| FEAT_SHA3 / SHA512 / BF16 / JSCVT | `hw.optional.arm.FEAT_SHA3` etc. | no PF_ flag; assumed present (Oryon, N2 have them) |
| FEAT_SVE / SVE2 | `hw.optional.arm.FEAT_SVE` (absent on Apple) | `PF_ARM_SVE_INSTRUCTIONS_AVAILABLE` (46) / `PF_ARM_SVE2_…` (47) |
| FEAT_SME | `hw.optional.arm.FEAT_SME` (M4+) | no PF_ flag; assumed absent |
| AES/Crypto | universal on all targets | assume true |
| FEAT_DOTPROD | universal on all targets | assume true |
