// gen_branch.cpp
// Branch prediction microbenchmark generator.
//
// ── Design notes ─────────────────────────────────────────────────────────────
//
// RSB DEPTH SWEEP
//   The Return Stack Buffer (RSB) is a hardware stack that shadows the
//   software call stack. Every BL pushes the return address; every RET pops
//   and uses it for speculative execution without waiting for memory. When
//   call depth exceeds RSB capacity, the hardware falls back to the BTB or
//   indirect predictor for the overflowed returns, paying a misprediction
//   penalty per miss.
//
//   We generate one JIT function per depth N. Each function:
//     outer_loop:
//       BL level_1
//       SUB/CBNZ (loop control)
//
//     level_k (k < N):              level_N:
//       SUB sp, sp, #16               RET
//       STR x30, [sp]
//       BL level_{k+1}
//       LDR x30, [sp]
//       ADD sp, sp, #16
//       RET
//
//   RSB usage per outer iteration: exactly N entries (one per BL in the chain,
//   including the outer loop's BL to level_1).
//
//   When N ≤ RSB_size: all N RET instructions are RSB-predicted → low latency.
//   When N > RSB_size: the (N - RSB_size) outermost returns are not in the
//   RSB and must be predicted by the BTB → higher latency per pair.
//
//   The latency per BL+RET pair (min_ns_per_insn from the harness, with
//   instructions_per_loop = N) reveals the per-pair cost. The first depth at
//   which this value jumps identifies the RSB capacity.
//
// INDIRECT BRANCH PREDICTOR CAPACITY SWEEP
//   We generate N tiny "trampoline" functions (each just RET) and a cycling
//   address table. The test loop:
//       AND  x21, x21, #(kTableSize-1)   // wrap index
//       LSL  x1,  x21, #3
//       LDR  x0,  [x20, x1]              // load next target address
//       BLR  x0                           // indirect call
//       ADD  x21, x21, #1
//       SUB  x19, x19, #1
//       CBNZ x19, loop_top
//
//   The address table is pre-expanded to kTableSize entries (cycling mod N),
//   so the data access pattern is always sequential (prefetchable), isolating
//   branch prediction from data cache effects.
//
//   When N=1: only one target → always predicted → fast.
//   When N is small: predictor learns the cycling pattern → fast.
//   When N exceeds predictor capacity: some entries are evicted and
//   mispredictions occur → latency rises.
//
//   KEY INTERPRETATION: the latency includes BOTH the BLR execution cost and
//   any misprediction penalty. With N=1 as baseline (always predicted), the
//   additional latency per BLR at each N reveals misprediction rate × penalty.
//
// CONDITIONAL BRANCH THROUGHPUT
//   Three instruction patterns compared:
//
//   1. CBNZ always-not-taken: multiple CBNZ instructions per iteration where
//      the register is guaranteed non-zero (register = 1, branch target is a
//      "skip" label that falls through anyway). Measures how many conditional
//      branches can be decoded and executed per cycle when all are predicted
//      not-taken with certainty.
//
//   2. TBZ bit0 alternating: test bit 0 of a counter that increments each
//      outer iteration. Bit 0 alternates 0,1,0,1... so the branch alternates
//      taken/not-taken. A modern predictor implementing pattern history should
//      predict this; a simple 1-bit saturating counter will mispredict half
//      the time. The difference from case 1 reveals prediction-table quality.
//
//   3. CBNZ always-taken: same instruction but the register is always zero.
//      (We zero x0 and CBNZ against it, so it always branches — to the next
//      loop iteration effectively.) Compares taken vs not-taken throughput.

#include "gen_branch.h"
#include "jit_buffer.h"
#include "harness.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstdlib>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Constants ─────────────────────────────────────────────────────────────────

// RSB sweep: depths to test. Covers likely RSB sizes of all target platforms:
//   Apple M1–M4 Firestorm:  ~50 entries  (Apple doesn't publish)
//   Cortex-A76/A78:          16 entries
//   Cortex-X1/X2/X3:         16–32 entries
//   Snapdragon Oryon:        ~48 entries (estimated)
//   Cortex-A55:               8 entries
static const uint32_t kRsbDepths[] = {
     1,  2,  3,  4,  5,  6,  7,  8,
    10, 12, 14, 16, 20, 24, 28, 32,
    36, 40, 44, 48, 52, 56, 60, 64,
};
static constexpr uint32_t kNumRsbDepths =
    static_cast<uint32_t>(sizeof(kRsbDepths) / sizeof(kRsbDepths[0]));

// Indirect predictor sweep: numbers of distinct targets to cycle through.
// 1 = always-same (baseline), up to 1024 (likely beyond any predictor).
static const uint32_t kIndTargetCounts[] = {
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32,
    48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
};
static constexpr uint32_t kNumIndTargetCounts =
    static_cast<uint32_t>(sizeof(kIndTargetCounts) / sizeof(kIndTargetCounts[0]));

// Address table for the indirect branch test: pre-expanded cycling array.
// Must be a power of 2 for the AND-mask wrapping trick.
// 4096 entries × 8 bytes = 32KB — fits in L1 on all targets.
static constexpr uint32_t kIndTableSize   = 4096;
static constexpr uint32_t kIndTableMask   = kIndTableSize - 1;
static constexpr uint32_t kIndTableBytes  = kIndTableSize * sizeof(uintptr_t);

// ── Loop count helpers ────────────────────────────────────────────────────────
//
// RSB: each outer iteration does N serial BL+RET pairs. At 2 cycles/pair
// and 3GHz, N=1 → 0.63ns/iter and N=64 → 40ns/iter. We target ~100ms.
// Scale: loops = ceil(100ms / (N × 2cycles / 3GHz)) = ceil(150M / N).
// Floor at 500K (avoids excessive JIT overhead per invocation).

static uint64_t rsb_loops_for_depth(uint32_t depth) {
    const uint64_t target = 150'000'000ULL;  // loop-iters × pairs to fill ~100ms
    const uint64_t loops  = target / depth;
    return (loops < 500'000) ? 500'000 : loops;
}

// Indirect: each BLR takes ~1ns when predicted, up to ~20ns when mispredicted.
// 2M iterations at 1ns = 2ms (too short for shallow N). Scale similarly.
// Target 100ms: loops = 100ms / (1ns * 1 BLR) = 100M for N=1 (overlong).
// Use a fixed 5M for all N — samples are 5-50ms. Acceptable: we have 7 samples.

static constexpr uint64_t kIndLoops = 5'000'000;

// ── RSB depth test builder ────────────────────────────────────────────────────

static JitPool::TestFn build_rsb_chain(uint32_t depth, uint64_t loops) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Pre-create labels for each level: level_labels[k] is the entry of
    // "level k+1" (the function called at the kth call-depth from the loop).
    // Using a VLA-equivalent: fixed maximum size, only [0..depth-1] are used.
    static constexpr uint32_t kMaxDepth = 128;
    Label level_labels[kMaxDepth];
    for (uint32_t k = 0; k < depth && k < kMaxDepth; ++k)
        level_labels[k] = a.new_label();

    Label loop_top = a.new_label();

    // ── Outer loop ────────────────────────────────────────────────────────
    // Prologue: save x19 (loop counter) and x30 (LR to harness).
    // 16-byte frame; stack remains 16-byte aligned.
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x30, ptr(sp));
    a.mov(x19, Imm(loops));

    // Align loop top to a cache-line boundary.
    a.align(AlignMode::kCode, 64);
    a.bind(loop_top);

    // Call into the chain. This BL is the first of the N BL+RET pairs
    // measured per outer iteration — it counts toward the RSB usage.
    a.bl(level_labels[0]);

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    // Epilogue.
    a.ldp(x19, x30, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    // ── Level functions ───────────────────────────────────────────────────
    // Each level is a small function. Intermediate levels must save/restore
    // x30 (LR) because they call further into the chain (non-leaf functions).
    // The innermost level (depth-1) is a leaf: just RET.
    //
    // Stack alignment per level: each sub sp, sp, #16 preserves 16-byte
    // alignment. At depth=64, peak stack usage = 64 × 16 = 1024 bytes,
    // well within thread stack limits.
    //
    // Why save only x30 (LR) and not the whole frame?
    //   These are microbenchmark-only functions with no local variables.
    //   Only LR needs preserving; the 8-byte padding word keeps sp aligned.
    for (uint32_t k = 0; k < depth; ++k) {
        a.bind(level_labels[k]);

        if (k == depth - 1) {
            // Innermost level: leaf function, no need to save LR.
            a.ret(x30);
        } else {
            // Non-leaf: save LR, call next level, restore LR, return.
            a.sub(sp, sp, Imm(16));
            a.str(x30, ptr(sp));         // save LR at [sp], [sp+8] is padding
            a.bl(level_labels[k + 1]);   // push to RSB, call next
            a.ldr(x30, ptr(sp));         // restore LR (if RSB predicted, never used)
            a.add(sp, sp, Imm(16));
            a.ret(x30);                  // RSB-predicted return
        }
    }

    return g_jit_pool->compile(code);
}

// ── Indirect branch predictor builder ────────────────────────────────────────

// Build N tiny trampoline functions (each just RET) and return their addresses
// in caller-allocated out_addrs[0..n_targets). Returns false on failure.
static bool build_trampolines(uint32_t n_targets,
                              JitPool::TestFn* out_fns,
                              uintptr_t* out_addrs) {
    for (uint32_t i = 0; i < n_targets; ++i) {
        CodeHolder code;
        g_jit_pool->init_code_holder(code);
        a64::Assembler a(&code);

        // Each trampoline is a bare RET. When called via BLR, x30 holds
        // the return address set by BLR; RET returns to it. No registers
        // are touched. This is a valid leaf function under any C ABI.
        a.ret(x30);

        out_fns[i] = g_jit_pool->compile(code);
        if (!out_fns[i]) return false;
        out_addrs[i] = reinterpret_cast<uintptr_t>(out_fns[i]);
    }
    return true;
}

// Build a test loop that BLRs to addresses cycling through n_targets entries
// in table[0..kIndTableSize). The table is pre-expanded (filled by the caller).
//
// Generated loop (pseudocode):
//   x20 = table_base  (baked immediate)
//   x19 = loops       (baked immediate)
//   x21 = 0           (current table index, wraps mod kIndTableSize via AND mask)
//
//   align 64
//   loop_top:
//     and  x1, x21, #kIndTableMask
//     lsl  x1, x1, #3
//     ldr  x0, [x20, x1]          // load target address
//     blr  x0                      // indirect call (indirect branch pred stresses here)
//     add  x21, x21, #1
//     sub  x19, x19, #1
//     cbnz x19, loop_top
//
// Note: x30 is set by BLR to point at the ADD x21 instruction. The trampoline
// RET returns there. This means x30 is safe to not save in our outer loop:
// BLR overwrites x30, the trampoline returns to the ADD, and the loop proceeds.
// The outer loop's own return address (to the harness) is saved in the prologue
// and restored in the epilogue before the final RET.

static JitPool::TestFn build_indirect_pred_loop(uintptr_t table_base,
                                                 uint64_t loops) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Prologue: save x19 (loop counter), x20 (table base), x21 (index), x30 (LR).
    // 32-byte frame: stp x19,x20 + stp x21,x30.
    a.sub(sp, sp, Imm(32));
    a.stp(x19, x20, ptr(sp));
    a.stp(x21, x30, ptr(sp, 16));

    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(static_cast<uint64_t>(table_base)));
    a.mov(x21, Imm(0));  // start at index 0

    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // Compute byte offset into table, wrapping via bitmask.
    // AND is single-cycle latency; LSL folds into the LDR addressing.
    a.and_(x1, x21, Imm(kIndTableMask));   // x1 = x21 & mask
    a.lsl(x1, x1, Imm(3));                 // x1 = x1 * 8 (pointer size)
    a.ldr(x0, ptr(x20, x1));               // x0 = table[x1/8]
    a.blr(x0);                             // indirect call — this is what we measure

    // BLR overwrites x30 with the return address (this ADD instruction).
    // The trampoline RET returns here. We don't need to save/restore x30
    // inside the loop — but we DO need the prologue-saved x30 for the
    // final epilogue RET, which is why we saved it above.
    a.add(x21, x21, Imm(1));
    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    // Epilogue: restore all callee-saved registers including x30 (harness LR).
    a.ldp(x19, x20, ptr(sp));
    a.ldp(x21, x30, ptr(sp, 16));
    a.add(sp, sp, Imm(32));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Conditional branch throughput builder ────────────────────────────────────
//
// Generates a loop with `unroll` copies of a conditional branch per iteration.
//
// CRITICAL DESIGN CHOICE — per-copy labels:
//   Each branch copy gets its own label bound immediately after it. For a
//   "taken" branch, this means the branch jumps to PC+4 (the next instruction).
//   For a "not-taken" branch, the label exists but is never jumped to.
//
//   This ensures ALL `unroll` copies execute every outer iteration, regardless
//   of taken/not-taken state. A shared label at the end would let an
//   always-taken first copy skip the remaining 15 copies, ruining the count.
//
//   A branch to the very next instruction (offset=+4) is valid ARM64 encoding.
//   The CPU still classifies it, looks it up in the prediction table, and
//   (for taken) redirects fetch to PC+4 — indistinguishable from a taken
//   branch with a non-trivial target in terms of predictor throughput.
//
// Callback signature: (a64::Assembler& a, uint32_t unroll_iter)
// Each invocation emits exactly one branch instruction and creates its own
// local label for the branch target, bound immediately after the instruction.
//
// Registers:
//   x19 = loop outer counter
//   x0  = constant 1 (CBZ x0 → never taken; CBNZ x0 → always taken)
//   x1  = iteration counter (bit 0 alternates 0,1,0,1... for TBZ tests)

template<typename F>
static JitPool::TestFn build_cond_branch_loop(uint64_t loops, uint32_t unroll,
                                               F&& emit_body) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(32));
    a.stp(x19, x20, ptr(sp));
    a.stp(x21, x30, ptr(sp, 16));

    a.mov(x19, Imm(loops));
    a.mov(x0,  Imm(1));   // constant non-zero for CBZ/CBNZ tests
    a.mov(x1,  Imm(0));   // iteration counter for TBZ alternating tests

    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // Each call to emit_body emits one branch + binds its own local skip label.
    for (uint32_t u = 0; u < unroll; ++u)
        emit_body(a, u);

    // Advance iteration counter once per outer iteration.
    a.add(x1, x1, Imm(1));

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    a.ldp(x19, x20, ptr(sp));
    a.ldp(x21, x30, ptr(sp, 16));
    a.add(sp, sp, Imm(32));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Benchmark runner helpers ──────────────────────────────────────────────────

static void run_one(const char* name, JitPool::TestFn fn,
                    const BenchmarkParams& params) {
    if (!fn) return;
    benchmark(fn, name, params);
    g_jit_pool->release(fn);
}

// ════════════════════════════════════════════════════════════════════════════
// Section 1: RSB depth sweep
// ════════════════════════════════════════════════════════════════════════════

static void run_rsb_tests(const BenchmarkParams& base) {
    printf("\n── RSB (Return Stack Buffer) depth sweep ───────────────────────\n");
    printf("  min_ns and clk are per BL+RET pair.\n"
           "  A latency jump identifies the RSB capacity.\n\n");

    double prev_clk = 0.0;

    for (uint32_t di = 0; di < kNumRsbDepths; ++di) {
        const uint32_t depth = kRsbDepths[di];
        const uint64_t loops = rsb_loops_for_depth(depth);

        JitPool::TestFn fn = build_rsb_chain(depth, loops);
        if (!fn) continue;

        BenchmarkParams p       = base;
        p.loops                 = loops;
        p.instructions_per_loop = depth;   // N BL+RET pairs per outer iteration
        p.bytes_per_insn        = 0;

        char name[48];
        snprintf(name, sizeof(name), "RSB depth %2u", depth);

        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        // Annotate when per-pair latency jumps — indicates RSB capacity exceeded.
        // A jump of ≥20% vs previous depth is considered significant.
        if (prev_clk > 0.0 && r.min_clocks_per_insn > prev_clk * 1.20) {
            printf("  ↑ RSB capacity likely exceeded at depth %u "
                   "(%.2f → %.2f clk/pair, +%.0f%%)\n",
                   depth,
                   prev_clk,
                   r.min_clocks_per_insn,
                   (r.min_clocks_per_insn / prev_clk - 1.0) * 100.0);
        }
        prev_clk = r.min_clocks_per_insn;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: Indirect branch predictor capacity sweep
// ════════════════════════════════════════════════════════════════════════════

static void run_indirect_pred_tests(const BenchmarkParams& base) {
    printf("\n── Indirect branch predictor capacity (BLR cycling N targets) ──\n");
    printf("  min_ns and clk are per BLR instruction.\n"
           "  Rising latency indicates predictor capacity exceeded.\n\n");

    // Allocate the cycling address table (heap, outside timed region).
    uintptr_t* table = static_cast<uintptr_t*>(malloc(kIndTableBytes));
    if (!table) {
        fprintf(stderr, "run_indirect_pred_tests: malloc failed\n");
        return;
    }

    // We'll compile up to kMaxTargets trampolines and reuse them as needed.
    // The largest sweep entry is kIndTargetCounts[last] = 1024.
    static constexpr uint32_t kMaxTargets = 1024;
    JitPool::TestFn trampoline_fns[kMaxTargets]  = {};
    uintptr_t       trampoline_addrs[kMaxTargets] = {};

    // Compile all trampolines upfront. They stay alive until the end of this
    // function. For small N, only the first N trampolines are used.
    // (We build all kMaxTargets to avoid repeated compile/release churn.)
    if (!build_trampolines(kMaxTargets, trampoline_fns, trampoline_addrs)) {
        fprintf(stderr, "run_indirect_pred_tests: trampoline compile failed\n");
        free(table);
        return;
    }

    double prev_clk = 0.0;

    for (uint32_t ti = 0; ti < kNumIndTargetCounts; ++ti) {
        const uint32_t n = kIndTargetCounts[ti];

        // Fill the cycling table: table[i] = trampoline_addrs[i % n].
        // Sequential cycling order — the predictor could learn this pattern
        // if N is small enough. That's intentional: we want to know the
        // capacity of the predictor working on a *learnable* pattern.
        for (uint32_t i = 0; i < kIndTableSize; ++i)
            table[i] = trampoline_addrs[i % n];

        JitPool::TestFn fn = build_indirect_pred_loop(
            reinterpret_cast<uintptr_t>(table), kIndLoops);
        if (!fn) continue;

        BenchmarkParams p       = base;
        p.loops                 = kIndLoops;
        p.instructions_per_loop = 1;   // one BLR per outer iteration
        p.bytes_per_insn        = 0;

        char name[48];
        snprintf(name, sizeof(name), "indirect BLR %4u targets", n);

        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        // Annotate capacity threshold: ≥15% latency jump vs previous count.
        if (prev_clk > 0.0 && r.min_clocks_per_insn > prev_clk * 1.15) {
            printf("  ↑ predictor capacity likely exceeded between %u and %u "
                   "targets (%.2f → %.2f clk/BLR)\n",
                   kIndTargetCounts[ti - 1], n,
                   prev_clk, r.min_clocks_per_insn);
        }
        prev_clk = r.min_clocks_per_insn;
    }

    // Release all trampolines.
    for (uint32_t i = 0; i < kMaxTargets; ++i)
        if (trampoline_fns[i]) g_jit_pool->release(trampoline_fns[i]);

    free(table);
}

// ════════════════════════════════════════════════════════════════════════════
// Section 3: Conditional branch throughput
// ════════════════════════════════════════════════════════════════════════════
//
// ARM64 branch instruction semantics used here:
//   CBZ  Xn, target : branches if Xn == 0
//   CBNZ Xn, target : branches if Xn != 0
//   TBZ  Xn, #bit, target : branches if bit N of Xn == 0
//   TBNZ Xn, #bit, target : branches if bit N of Xn != 0
//
// With x0 = 1 (constant non-zero):
//   CBZ  x0 → NEVER   taken (x0 != 0, CBZ only branches when zero)
//   CBNZ x0 → ALWAYS  taken (x0 != 0)
//
// With x1 = per-outer-iteration counter starting at 0:
//   TBZ  x1, #0 → TAKEN when x1 is even (bit0==0), NOT-TAKEN when odd
//   TBNZ x1, #0 → TAKEN when x1 is odd  (bit0==1), NOT-TAKEN when even
//
// Unroll=16: dilutes loop-control overhead to <7%, fits in ≤2 cache lines.

static void run_cond_branch_tests(const BenchmarkParams& base) {
    printf("\n── Conditional branch throughput ───────────────────────────────\n");

    const uint64_t loops  = base.loops;
    const uint32_t unroll = 16;
    char name[64];

    BenchmarkParams p       = base;
    p.instructions_per_loop = unroll;
    p.bytes_per_insn        = 0;

    // ── CBZ never-taken ───────────────────────────────────────────────────
    // x0=1. CBZ branches only if zero → never fires. Each copy emits a local
    // label bound to PC+4 (the next instruction). Baseline: perfect not-taken
    // prediction. Measures decode/execute throughput ceiling for conditional
    // branches with zero mispredictions.
    {
        auto fn = build_cond_branch_loop(loops, unroll,
            [](a64::Assembler& a, uint32_t) {
                Label sk = a.new_label();
                a.cbz(x0, sk);  // x0=1 → bit0=1 → CBZ(zero) → never taken
                a.bind(sk);
            });
        snprintf(name, sizeof(name), "CBZ  never-taken      (%ux unroll)", unroll);
        run_one(name, fn, p);
    }

    // ── CBNZ always-taken ─────────────────────────────────────────────────
    // x0=1. CBNZ branches if non-zero → always fires. Branch target is the
    // label bound at PC+4 (next instruction), so no net change in fetch flow.
    // Compares with CBZ-never-taken to reveal taken vs not-taken asymmetry
    // in the predictor or branch execution unit.
    {
        auto fn = build_cond_branch_loop(loops, unroll,
            [](a64::Assembler& a, uint32_t) {
                Label sk = a.new_label();
                a.cbnz(x0, sk); // x0=1 → non-zero → always taken (to PC+4)
                a.bind(sk);
            });
        snprintf(name, sizeof(name), "CBNZ always-taken     (%ux unroll)", unroll);
        run_one(name, fn, p);
    }

    // ── TBZ bit0 alternating ──────────────────────────────────────────────
    // x1 increments once per outer iteration. Bit 0 of x1 alternates 0,1,0,1.
    // TBZ branches on bit0==0 → taken on EVEN iterations, not-taken on ODD.
    // All 16 unroll copies see the SAME x1 in a single iteration (x1 advances
    // after the body), so all 16 agree — testing iteration-level alternation.
    //
    // Expected outcomes:
    //   ≈ CBZ-never-taken clk → predictor handles TNTN pattern (≥1-bit history)
    //   >> CBZ-never-taken    → predictor can't learn this; ~50% misprediction
    {
        auto fn = build_cond_branch_loop(loops, unroll,
            [](a64::Assembler& a, uint32_t) {
                Label sk = a.new_label();
                a.tbz(x1, Imm(0), sk);  // taken if bit0==0 (even iters)
                a.bind(sk);
            });
        snprintf(name, sizeof(name), "TBZ  bit0 alternating (%ux unroll)", unroll);
        run_one(name, fn, p);
    }

    // ── TBNZ bit0 alternating ─────────────────────────────────────────────
    // Complement of TBZ above: taken on ODD iterations instead of EVEN.
    // Confirms the result is symmetric and that TBZ/TBNZ at the same PC
    // share the same prediction table entry (should give identical results).
    {
        auto fn = build_cond_branch_loop(loops, unroll,
            [](a64::Assembler& a, uint32_t) {
                Label sk = a.new_label();
                a.tbnz(x1, Imm(0), sk); // taken if bit0==1 (odd iters)
                a.bind(sk);
            });
        snprintf(name, sizeof(name), "TBNZ bit0 alternating (%ux unroll)", unroll);
        run_one(name, fn, p);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_branch_tests(const BenchmarkParams& base_params) {
    run_rsb_tests(base_params);
    run_indirect_pred_tests(base_params);
    run_cond_branch_tests(base_params);
}

} // namespace arm64bench::gen
