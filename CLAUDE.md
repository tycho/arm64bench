# arm64bench

Precision microarchitecture benchmarking framework for ARM64/AArch64 processors. Measures CPU performance across integer ALU, memory hierarchy, branch prediction, and FP/NEON SIMD. Targets Apple Silicon (macOS), Windows ARM64, and Linux AArch64.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
ninja
```

**Requirements:** CMake 3.20+, Clang (MSVC is explicitly rejected). AsmJit is fetched automatically as a dependency.

Default build type is `RelWithDebInfo`. Release builds use `-O3`.

Run with `sudo ./arm64bench` on macOS 15+ (Sequoia/Tahoe) to enable hardware PMU cycle counting; unprivileged runs fall back to Tier 2 ratio normalization.

## Run

```bash
./arm64bench [--all | --integer | --memory | --branch | --simd | --pitfalls]
             [--MHz <freq>] [--samples <n>] [--warmup <n>] [--csv]
```

Default (no flags): runs integer and memory tests.

## Architecture

| File | Purpose |
|------|---------|
| `src/main.cpp` | CLI entry point |
| `src/timer.h/.cpp` | Platform-agnostic monotonic timer |
| `src/harness.h/.cpp` | Benchmark runner and statistical analysis |
| `src/cycle_counter.h/.cpp` | PMU cycle counter abstraction (macOS kpc, Windows PMCCNTR_EL0) |
| `src/jit_buffer.h/.cpp` | JIT memory pool with W^X handling |
| `src/calibrate.cpp` | CPU frequency estimation |
| `src/gen_integer.h/.cpp` | Integer ALU latency/throughput tests |
| `src/gen_memory.h/.cpp` | Cache/memory hierarchy tests |
| `src/gen_branch.h/.cpp` | Branch prediction tests |
| `src/gen_fp_simd.h/.cpp` | FP, NEON SIMD, crypto, and advanced SIMD tests |
| `src/gen_lse.h/.cpp` | LSE atomics latency/throughput tests |
| `src/gen_pitfalls.h/.cpp` | Micro-architectural pathology tests (barriers, LRCPC, store forwarding) |

## Code Conventions

- **Namespace:** `arm64bench` (generators in `arm64bench::gen`)
- **Naming:** `PascalCase` for types, `snake_case` for functions/variables, `g_` prefix for globals
- **Headers:** `#pragma once`
- **No exceptions, no RTTI** (`-fno-exceptions -fno-rtti`)
- Section separators: `// ── Description ──────────────────────`
- Platform guards: `#if defined(_WIN32)`, `#ifdef __APPLE__`, `#ifdef __linux__`

### JIT and compile-time CPU feature macros

All test code is JIT-emitted via AsmJit. The host compiler only sees C++ method calls like `a.usdot(...)` — it never emits the target instruction itself. Therefore **compile-time feature macros (`__ARM_FEATURE_CRYPTO`, `__ARM_FEATURE_I8MM`, etc.) are never needed** to guard JIT test code. Use only runtime feature detection:

- **macOS**: `sysctlbyname("hw.optional.arm.FEAT_XXX", ...)` — comprehensive, reliable
- **Windows**: `IsProcessorFeaturePresent(PF_ARM_*)` — limited coverage; see notes per-feature
- **Linux**: `getauxval(AT_HWCAP)` / `AT_HWCAP2`

Feature detection functions follow the pattern in `gen_pitfalls.cpp::has_feat_lrcpc()`.

## Commit Discipline

Each commit should be one logical group of changes — typically one new test
section, or one focused refactor. The constraint: **every commit must
independently compile and run.** No commit should leave the tree in a
half-wired state (e.g., a function defined but not called, or a header
declaration without an implementation). When adding several related test
sections, split them into a series of small commits — one per section —
so any single one can be reverted cleanly without unwinding the rest.
Build (`ninja`) and run the affected `--integer` / `--simd` / `--pitfalls`
flag between each commit.

## JIT Loop Structure

All test generators follow this pattern:
- Save callee-saved registers; load loop counter and constant source into x20
- `SUB x19, x19, #1` + `CBNZ` for loop control (avoids writing condition flags)
- Align loop to cache line
- Unroll factor varies: higher for fast instructions (ADD), lower for slow (SDIV)

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

    ratio = test_ns / avg(ref_before_ns, ref_after_ns)

If `|ref_after - ref_before| / ref_before > threshold`, the measurement is flagged as potentially
affected by a P-state change *between* the reference probes (external throttle). The sample is
re-taken up to a retry limit.

**Limitation**: does not catch instruction-induced throttling (the test instructions change the
clock speed; both reference probes see the un-throttled rate). For instruction classes known to
risk this (SVE2 wide ops), results should be labelled as potentially reflecting throttled execution.

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

### AsmJit API notes

- `Gp` not `GpX` for general-purpose register arguments in helper functions
- `a.embed(&word, 4)` to hand-encode instructions not exposed in AsmJit's C++ API
  (used for LDAPR, LDAPUR, STLUR in `gen_pitfalls.cpp`)
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
| **LSE atomics** | `gen_lse.cpp` | LDADDAL=7 clk, SWPAL=2.5 clk, LDAXR+STLXR=11 clk |
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

## Planned Test Coverage

| Category | Tests | Notes |
|---|---|---|
| **UDIV throughput** | Independent UDIV chains | Latency known (~10 clk); throughput (units) unknown |
| **Prefetcher** | Stride sweep, descending scan, PRFM effectiveness | How far ahead does the hardware prefetcher reach? |
| **OOO window** | Dependent chain length × latency product → ROB depth | Complex to expose cleanly for ALU-only chains; requires careful design |
| **FEAT_LRCPC3** | LDIAPP / STILP pair instructions | Not present on any current Apple Silicon (M1–M5); available check via `hw.optional.arm.FEAT_LRCPC3` |
| **SVE2** | Wide vector ops (if present) | Not on Apple Silicon; check at runtime on Linux/Windows (Snapdragon X has SVE2). Results may require PMU (Tier 1) to be trustworthy — instruction-induced throttling risk |
| **SDOT/SMMLA cross-platform** | Compare MAC throughput on Snapdragon X | Does Oryon have dedicated SMMLA hardware, or also micro-op fusion like M5? |
| **FCVTL/FCVTN** | FP16↔FP32 conversion throughput | Widening/narrowing pipeline characterization |

## Feature Detection Reference

| Feature | macOS sysctl | Windows |
|---|---|---|
| FEAT_I8MM | `hw.optional.arm.FEAT_I8MM` | `PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE` (proxy; no direct PF_ exists) |
| FEAT_LRCPC | `hw.optional.arm.FEAT_LRCPC` | assume true (Oryon) |
| FEAT_LRCPC2 | `hw.optional.arm.FEAT_LRCPC2` | assume true (Oryon) |
| FEAT_LRCPC3 | `hw.optional.arm.FEAT_LRCPC3` | unknown |
| AES/Crypto | universal on all targets | assume true |
| FEAT_DOTPROD | universal on all targets | assume true |
