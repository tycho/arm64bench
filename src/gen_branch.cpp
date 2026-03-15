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
//   Apple M1–M4 Firestorm:  50 entries  (confirmed empirically — see results)
//   Cortex-A76/A78:          16 entries
//   Cortex-X1/X2/X3:         16–32 entries
//   Snapdragon Oryon:        ~48 entries (estimated)
//   Cortex-A55:               8 entries
//
// 49/50/51 are included to pinpoint the M1 boundary with single-entry precision.
static const uint32_t kRsbDepths[] = {
     1,  2,  3,  4,  5,  6,  7,  8,
    10, 12, 14, 16, 20, 24, 28, 32,
    36, 40, 44, 48, 49, 50, 51, 52, 56, 60, 64,
};
static constexpr uint32_t kNumRsbDepths =
    static_cast<uint32_t>(sizeof(kRsbDepths) / sizeof(kRsbDepths[0]));

// Indirect predictor CYCLING test: numbers of distinct targets to cycle through
// at a single BLR site. Tests misprediction penalty and cycling-pattern learning.
// 1 = always-same (baseline), up to 1024.
static const uint32_t kIndCycleCounts[] = {
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32,
    48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
};
static constexpr uint32_t kNumIndCycleCounts =
    static_cast<uint32_t>(sizeof(kIndCycleCounts) / sizeof(kIndCycleCounts[0]));

// Indirect predictor CAPACITY test: numbers of distinct BLR SITES, each always
// calling the same fixed target. Reveals OOO BLR throughput, not table capacity
// (since all sites share one target, the predictor needs only 1 entry).
static const uint32_t kIndSiteCounts[] = {
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64,
};
static constexpr uint32_t kNumIndSiteCounts =
    static_cast<uint32_t>(sizeof(kIndSiteCounts) / sizeof(kIndSiteCounts[0]));

// Indirect predictor UNIQUE-TARGET test: N BLR sites, N distinct targets.
// This forces the predictor to maintain N separate (site→target) entries
// simultaneously. Sweep to find where entries are evicted and latency rises.
// Capped at 256: MOV+BLR per site = 20 bytes, 256 sites = 5KB of loop body,
// still within prefetch/branch-predictor tracking budget.
static const uint32_t kIndUniqueSiteCounts[] = {
     1,  2,  4,  8,  12,  16,  24,  32,
    48, 64, 96, 128, 192, 256,
};
static constexpr uint32_t kNumIndUniqueSiteCounts =
    static_cast<uint32_t>(sizeof(kIndUniqueSiteCounts) / sizeof(kIndUniqueSiteCounts[0]));

// Address table for the cycling test: pre-expanded array, power-of-2 size
// for AND-mask wrapping. 4096 × 8B = 32KB — fits in L1 on all targets.
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

// Indirect cycling test: loop count per target count N.
// At N=1 (always predicted, ~3 clk/BLR at 3GHz): 5M loops = ~5ms — too short,
// causing the 12% CoV we observed. Scale up for small N.
// Target: ~50ms per sample = 50ms × 3GHz / 3 cycles = 50M loop-instructions.
// For the capacity test (N sites, all predicted): same target.

static uint64_t ind_cycle_loops_for_n(uint32_t /*n*/) {
    // All cycle test entries converge to ~10 clk (mispredicting), so they're
    // already ~16ms at 5M. Only N=1 is short (~5ms). Use 30M for all — it
    // keeps N=1 clean (~30ms) and N≥2 at ~100ms. Cap at 30M.
    return 30'000'000ULL;
}

static uint64_t ind_capacity_loops_for_n(uint32_t n) {
    // Each iteration executes N BLR instructions. Target ~50ms per sample.
    // At predicted ~3 clk/BLR: loops = 50ms × 3GHz / (3 × N) = 50M / N.
    const uint64_t loops = 50'000'000ULL / n;
    return (loops < 200'000) ? 200'000 : loops;
}

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

// ── Indirect predictor CAPACITY test builder ──────────────────────────────────
//
// This tests the predictor's (call-site → target) table capacity, which is a
// fundamentally different thing from the cycling test above.
//
// WHAT IT MEASURES:
//   N distinct BLR instructions at N distinct code addresses, ALL calling the
//   SAME target trampoline (address baked into x0 in the prologue). Each BLR
//   creates one (site_address → target) association in the predictor's table.
//
//   When N ≤ predictor capacity: all N associations fit, all BLRs are
//   predicted → latency ≈ same as a direct call (~3 clk/BLR).
//
//   When N > predictor capacity: some associations are evicted each time the
//   loop runs, so the evicted BLR sites mispredict → latency rises.
//
// WHY THIS IS DIFFERENT FROM THE CYCLING TEST:
//   Cycling test: 1 BLR site, N possible targets. Tests whether the predictor
//     can learn a repeating N-target pattern at a single site.
//     Result: no — the M1 predictor only tracks the last-seen target per site.
//     Mispredicts every BLR once N≥2, regardless of pattern length.
//
//   Capacity test: N BLR sites, 1 target. Tests how many (site→target) pairs
//     the predictor can hold simultaneously before evicting entries.
//
// GENERATED LOOP:
//   prologue: save x19 (counter), x30 (LR)
//   mov x0, #trampoline_addr   ← single target, never changes
//   loop_top:
//     BLR x0   ← site 1 at this PC
//     BLR x0   ← site 2 at PC+4
//     ...
//     BLR x0   ← site N at PC+(N-1)*4
//     sub x19, #1
//     cbnz x19, loop_top
//   epilogue
//
// BLR clobbers only x30 (= return address = instruction after the BLR).
// The trampoline immediately RETs to that return address. x0 is untouched.
// So x0 stays valid as the target address across all N BLR calls per iteration.

static JitPool::TestFn build_ind_capacity_loop(uintptr_t trampoline_addr,
                                                uint32_t  n_sites,
                                                uint64_t  loops) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // Minimal prologue: save only x19 (loop counter) and x30 (harness LR).
    // x0 is caller-saved and holds the target — no need to save it.
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x30, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x0, Imm(static_cast<uint64_t>(trampoline_addr)));

    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    // Emit n_sites BLR x0 instructions at n_sites distinct code addresses.
    // Each BLR sets x30 = (address of next instruction), trampoline RETs there.
    for (uint32_t s = 0; s < n_sites; ++s)
        a.blr(x0);

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    a.ldp(x19, x30, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    return g_jit_pool->compile(code);
}

// ── Indirect predictor unique-target test builder ─────────────────────────────
//
// N distinct BLR sites, each calling a DIFFERENT target trampoline.
// Every (site_PC, target_addr) pair is unique — the predictor must track all
// N associations simultaneously to avoid mispredictions.
//
// GENERATED LOOP:
//   loop_top:
//     mov x0, #addr_0    ← 1–4 instructions depending on address width
//     blr x0             ← site 0: predictor entry (loop_top+0, addr_0)
//     mov x0, #addr_1
//     blr x0             ← site 1: predictor entry (loop_top+20, addr_1)
//     ...
//     mov x0, #addr_{N-1}
//     blr x0             ← site N-1
//     sub x19, x19, #1
//     cbnz x19, loop_top
//
// WHY MOV IS SAFE HERE:
//   MOV x0 is a 1-cycle operation on all ARM64 cores. It does NOT create a
//   dependency between adjacent BLR instructions — x0 is written fresh before
//   each BLR, so the OOO engine can still look ahead and speculatively execute
//   the next BLR using the (predicted, not-yet-retired) result of the upcoming
//   MOV. The sequence is not serially dependent; it can pipeline freely.
//
//   The only true dependency chain is:
//     BLR site_k → trampoline RET → returns to MOV x0, #addr_{k+1} → BLR site_{k+1}
//   This is the same chain the 1-target test has, so the per-BLR overhead is
//   comparable. Any additional latency at large N comes from predictor misses.
//
// NOTE: AsmJit emits MOVZ + up to 3× MOVK for 64-bit immediates, so each
// "mov x0, #addr" is 1–4 instructions. The harness normalises by N (BLR count
// only), so the MOV overhead inflates the raw elapsed time but NOT the reported
// clk/BLR — the per-BLR number is derived from loops×N total_BLRs, not
// loops×N×(MOV_count+1). This is intentional: we want clk/BLR to reflect
// the cost attributable to branch prediction.

static uint64_t ind_unique_loops_for_n(uint32_t n) {
    // Target ~50ms per sample. Each iteration: N BLR calls, each ~3–10 clk.
    // Conservative estimate: 10 clk/BLR → loops = 50ms×3GHz / (10×N) = 15M/N.
    const uint64_t loops = 15'000'000ULL / n;
    return (loops < 100'000) ? 100'000 : loops;
}

static JitPool::TestFn build_ind_unique_target_loop(
        const uintptr_t* addrs,  // addrs[0..n_sites)
        uint32_t         n_sites,
        uint64_t         loops) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(16));
    a.stp(x19, x30, ptr(sp));

    a.mov(x19, Imm(loops));

    a.align(AlignMode::kCode, 64);

    Label loop_top = a.new_label();
    a.bind(loop_top);

    for (uint32_t s = 0; s < n_sites; ++s) {
        // Load the unique target for site s into x0.
        // BLR x0 then executes at a distinct PC each iteration.
        a.mov(x0, Imm(static_cast<uint64_t>(addrs[s])));
        a.blr(x0);
    }

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, loop_top);

    a.ldp(x19, x30, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    return g_jit_pool->compile(code);
}
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
    // ── Cycling pattern test (misprediction penalty probe) ─────────────────
    // One BLR site cycling through N distinct targets in round-robin order.
    // Measures: can the predictor learn a repeating N-target cycle?
    // Result on M1: No — latency jumps immediately at N=2 and stays flat,
    // indicating the predictor only tracks the last-seen target per site.
    // The flat ~10 clk from N=2 to N=1024 is the raw misprediction penalty.
    printf("\n── BLR cycling N targets (misprediction penalty probe) ─────────\n");
    printf("  One BLR site cycling round-robin through N targets.\n"
           "  Flat latency ≥ N=2 = predictor cannot learn cycling patterns.\n"
           "  The N=1 latency is the predicted baseline; N=2 reveals penalty.\n\n");

    uintptr_t* table = static_cast<uintptr_t*>(malloc(kIndTableBytes));
    if (!table) {
        fprintf(stderr, "run_indirect_pred_tests: malloc failed\n");
        return;
    }

    static constexpr uint32_t kMaxTargets = 1024;
    JitPool::TestFn trampoline_fns[kMaxTargets]  = {};
    uintptr_t       trampoline_addrs[kMaxTargets] = {};

    if (!build_trampolines(kMaxTargets, trampoline_fns, trampoline_addrs)) {
        fprintf(stderr, "run_indirect_pred_tests: trampoline compile failed\n");
        free(table);
        return;
    }

    double prev_clk = 0.0;

    for (uint32_t ti = 0; ti < kNumIndCycleCounts; ++ti) {
        const uint32_t n     = kIndCycleCounts[ti];
        const uint64_t loops = ind_cycle_loops_for_n(n);

        for (uint32_t i = 0; i < kIndTableSize; ++i)
            table[i] = trampoline_addrs[i % n];

        JitPool::TestFn fn = build_indirect_pred_loop(
            reinterpret_cast<uintptr_t>(table), loops);
        if (!fn) continue;

        BenchmarkParams p       = base;
        p.loops                 = loops;
        p.instructions_per_loop = 1;
        p.bytes_per_insn        = 0;

        char name[56];
        snprintf(name, sizeof(name), "BLR cycling %4u targets", n);

        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        if (prev_clk > 0.0 && r.min_clocks_per_insn > prev_clk * 1.15) {
            printf("  ↑ misprediction onset between %u and %u targets"
                   " (%.2f → %.2f clk/BLR)\n",
                   kIndCycleCounts[ti - 1], n,
                   prev_clk, r.min_clocks_per_insn);
        }
        prev_clk = r.min_clocks_per_insn;
    }

    for (uint32_t i = 0; i < kMaxTargets; ++i)
        if (trampoline_fns[i]) g_jit_pool->release(trampoline_fns[i]);

    free(table);
}

static void run_ind_capacity_tests(const BenchmarkParams& base) {
    // ── BLR throughput (single target, multiple sites) ─────────────────────
    // N BLR instructions per loop iteration, ALL calling the same target.
    // Since every site predicts to the same address, only ONE predictor entry
    // is ever needed — this is NOT a capacity test. Instead it reveals:
    //   - The OOO engine's BLR throughput ceiling (how many can overlap)
    //   - The RSB/return-prediction mechanism for non-varying targets
    // The per-BLR latency decreases as N grows because the OOO can pipeline
    // more BLR+RET pairs in flight simultaneously.
    printf("\n── BLR N sites → 1 target (OOO throughput, all same target) ────\n");
    printf("  Decreasing clk/BLR with N = OOO pipelining BLR+RET pairs.\n"
           "  Not a capacity test — all sites predict to the same target.\n\n");

    // Build a single target trampoline.
    JitPool::TestFn trampoline_fn   = nullptr;
    uintptr_t       trampoline_addr = 0;
    if (!build_trampolines(1, &trampoline_fn, &trampoline_addr)) {
        fprintf(stderr, "run_ind_capacity_tests: trampoline compile failed\n");
        return;
    }

    double prev_clk = 0.0;

    for (uint32_t si = 0; si < kNumIndSiteCounts; ++si) {
        const uint32_t n     = kIndSiteCounts[si];
        const uint64_t loops = ind_capacity_loops_for_n(n);

        JitPool::TestFn fn = build_ind_capacity_loop(trampoline_addr, n, loops);
        if (!fn) continue;

        BenchmarkParams p       = base;
        p.loops                 = loops;
        p.instructions_per_loop = n;  // N BLR instructions per loop iteration
        p.bytes_per_insn        = 0;

        char name[56];
        snprintf(name, sizeof(name), "BLR %2u sites → 1 target", n);

        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        // Flag when per-BLR latency rises ≥15% vs previous site count.
        if (prev_clk > 0.0 && r.min_clocks_per_insn > prev_clk * 1.15) {
            printf("  ↑ predictor capacity likely exceeded between %u and %u "
                   "sites (%.2f → %.2f clk/BLR)\n",
                   kIndSiteCounts[si - 1], n,
                   prev_clk, r.min_clocks_per_insn);
        }
        prev_clk = r.min_clocks_per_insn;
    }

    if (trampoline_fn) g_jit_pool->release(trampoline_fn);
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

static void run_ind_unique_target_tests(const BenchmarkParams& base) {
    // ── BLR N sites → N unique targets (genuine predictor capacity probe) ──
    // Each of the N BLR instructions has a DISTINCT target trampoline.
    // The predictor must maintain N independent (site→target) entries.
    //
    // Interpretation:
    //   Flat clk/BLR across all N → predictor holds all entries; no evictions.
    //   Rising clk/BLR at some N* → evictions begin; N* is a lower bound on
    //     the predictor's (indirect branch site → target) table capacity.
    //
    // IMPORTANT: As N grows, the loop body grows too (N × MOV+BLR ≈ N×20B).
    // At N=256 the loop body is ~5KB — still within I-cache but may stress
    // the loop buffer. Any CoV spike at large N may reflect I-cache pressure
    // rather than predictor pressure. The annotation threshold is conservative.
    printf("\n── BLR N sites → N unique targets (predictor capacity probe) ───\n");
    printf("  N BLR sites each with a distinct target trampoline.\n"
           "  Flat clk/BLR = predictor holds all N entries.\n"
           "  Rising clk/BLR = evictions beginning (capacity exceeded).\n\n");

    // Build all trampolines up front. kNumIndUniqueSiteCounts last entry is 256.
    static constexpr uint32_t kMaxUnique = 256;
    JitPool::TestFn trampoline_fns[kMaxUnique]  = {};
    uintptr_t       trampoline_addrs[kMaxUnique] = {};

    if (!build_trampolines(kMaxUnique, trampoline_fns, trampoline_addrs)) {
        fprintf(stderr, "run_ind_unique_target_tests: trampoline compile failed\n");
        return;
    }

    double prev_clk = 0.0;

    for (uint32_t si = 0; si < kNumIndUniqueSiteCounts; ++si) {
        const uint32_t n     = kIndUniqueSiteCounts[si];
        const uint64_t loops = ind_unique_loops_for_n(n);

        JitPool::TestFn fn = build_ind_unique_target_loop(
            trampoline_addrs, n, loops);
        if (!fn) continue;

        BenchmarkParams p       = base;
        p.loops                 = loops;
        p.instructions_per_loop = n;   // N BLR instructions per iteration
        p.bytes_per_insn        = 0;

        char name[64];
        snprintf(name, sizeof(name), "BLR %3u sites → %3u targets", n, n);

        const BenchmarkResult r = benchmark(fn, name, p);
        g_jit_pool->release(fn);

        if (prev_clk > 0.0 && r.min_clocks_per_insn > prev_clk * 1.15) {
            printf("  ↑ predictor capacity likely exceeded between %u and %u "
                   "sites (%.2f → %.2f clk/BLR)\n",
                   kIndUniqueSiteCounts[si - 1], n,
                   prev_clk, r.min_clocks_per_insn);
        }
        prev_clk = r.min_clocks_per_insn;
    }

    for (uint32_t i = 0; i < kMaxUnique; ++i)
        if (trampoline_fns[i]) g_jit_pool->release(trampoline_fns[i]);
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_branch_tests(const BenchmarkParams& base_params) {
    run_rsb_tests(base_params);
    run_indirect_pred_tests(base_params);
    run_ind_capacity_tests(base_params);
    run_ind_unique_target_tests(base_params);
    run_cond_branch_tests(base_params);
}

} // namespace arm64bench::gen
