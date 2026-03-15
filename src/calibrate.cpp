// calibrate.cpp
// CPU frequency calibration via a JIT-generated chained-ADD loop.
//
// ── Why a chained ADD loop works as a ruler ──────────────────────────────────
//
// The ARMv8 architecture specification guarantees ADD (shifted register and
// immediate forms) has a result latency of 1 cycle on every compliant
// implementation. This is not a microarchitecture detail that varies — it is
// an architectural commitment. Therefore:
//
//   elapsed_cycles = loops × unroll  (each chained ADD contributes 1 cycle)
//   freq_hz        = elapsed_cycles / elapsed_wall_seconds
//
// The loop control instructions (SUB x19 + CBNZ) operate on an independent
// register and are absorbed by the out-of-order engine in parallel with the
// ADD chain. They add negligible overhead. A large unroll factor (64 here)
// further minimizes any residual loop-control contribution.
//
// ── Why we take the maximum of multiple runs ─────────────────────────────────
//
// A run that is slowed by a scheduler preemption, thermal event, or P→E core
// migration will UNDERCOUNT the true frequency (numerator fixed, denominator
// too large). We take the minimum elapsed time (= maximum apparent frequency)
// across N runs because the fastest run is the one least contaminated by
// external interference. This is the same logic as taking the minimum sample
// in the benchmarking harness.
//
// ── Limitations ──────────────────────────────────────────────────────────────
//
// On platforms with dynamic frequency scaling (all of them), this measures
// the frequency at which the core was running during calibration, which may
// differ from the frequency during benchmarks. On macOS and Windows ARM64 we
// have no mechanism to lock the P-state. The best we can do is:
//   1. Run calibration warm (after a warm-up pass) so boost is engaged.
//   2. Run benchmarks at the same elevated priority as calibration.
//   3. Let the user override with --MHz if they know the exact frequency.
// A persistent frequency mismatch shows up as clock counts that are slightly
// off from integer values, which is visually obvious in the output.

#include "timer.h"
#include "jit_buffer.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>

namespace arm64bench {

using namespace asmjit;
using namespace asmjit::a64;

// Build a tight chained-ADD loop for calibration.
// The critical path is the single chain x0 = x0 + x1, iterated
// (loops × unroll) times, giving exactly that many cycles of execution.
static JitPool::TestFn build_calibration_fn(uint64_t loops, uint32_t unroll) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Save x19 (loop counter) and x20 (addend, so we don't disturb caller).
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(1));     // constant addend; x0 += 1 each step
    a.mov(x0,  Imm(0));     // chain register initial value

    // 64-byte cache-line alignment for consistent fetch behaviour.
    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // Unrolled chained ADD body.
    // All instructions read x0 and write x0: strictly serial, 1 cycle each.
    // x20 is read as the addend and is an architectural constant (never written
    // inside the loop), so it has no carry-over dependency between iterations.
    for (uint32_t u = 0; u < unroll; ++u)
        a.add(x0, x0, x20);

    // Loop control: independent of x0/x20 chain (uses x19 only).
    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn)
        fprintf(stderr, "calibrate: failed to compile calibration function\n");
    return fn;
}

uint64_t calibrate_cpu_freq() {
    if (!g_jit_pool) {
        fprintf(stderr, "calibrate_cpu_freq: g_jit_pool not initialized\n");
        return 0;
    }

    // Target ~200ms per run at the expected frequency range of 1–4 GHz.
    //
    // At 3 GHz with unroll=64:
    //   elapsed = (loops × 64) / 3e9 = 200ms  →  loops = 3e9 × 0.2 / 64 ≈ 9.4M
    //
    // 10M loops × 64 unroll = 640M cycles → ~213ms at 3 GHz.
    // At 1 GHz (worst case slow core): ~640ms per run, still acceptable.
    // At 4 GHz (best case boost): ~160ms per run.
    const uint32_t kUnroll = 64;
    const uint64_t kLoops  = 10'000'000;

    JitPool::TestFn fn = build_calibration_fn(kLoops, kUnroll);
    if (!fn) return 0;

    // Two untimed warm-up calls:
    //   First call: I-cache cold, branch predictor cold. Discard.
    //   Second call: everything warm. Discard — but now the CPU has sustained
    //   load for ~200ms, giving the frequency governor time to reach boost.
    fn();
    fn();

    // Timed runs. Take the maximum apparent frequency (minimum elapsed time).
    // 5 runs is enough to reliably get one sample uncontaminated by OS jitter.
    static constexpr int kRuns = 5;
    double best_freq_hz = 0.0;

    const uint64_t kTotalCycles = static_cast<uint64_t>(kLoops) * kUnroll;

    for (int i = 0; i < kRuns; ++i) {
        // Align to a timer tick boundary before each run, same as the harness,
        // so we don't accumulate partial-tick error across runs.
        const RawTick t0 = wait_for_tick();
        fn();
        const RawTick t1 = tick_now();

        const double elapsed_s = ticks_to_ns_f(t1 - t0) * 1e-9;
        if (elapsed_s <= 1e-6) continue; // implausibly short, skip

        // freq = total_cycles / elapsed_seconds
        // Each chained ADD = 1 architectural cycle, so total_cycles is exact.
        const double freq = static_cast<double>(kTotalCycles) / elapsed_s;
        if (freq > best_freq_hz)
            best_freq_hz = freq;
    }

    g_jit_pool->release(fn);

    // Plausibility guard: reject anything outside 100 MHz – 10 GHz.
    if (best_freq_hz < 100e6 || best_freq_hz > 10e9) {
        fprintf(stderr,
                "calibrate_cpu_freq: implausible result %.0f MHz -- "
                "use --MHz to set manually\n",
                best_freq_hz / 1e6);
        g_cpu_freq_hz = 0;
        return 0;
    }

    // Round to nearest Hz. The measurement noise floor is much larger than
    // 1 Hz, so this rounding is cosmetically fine.
    g_cpu_freq_hz = static_cast<uint64_t>(best_freq_hz + 0.5);
    return g_cpu_freq_hz;
}

} // namespace arm64bench
