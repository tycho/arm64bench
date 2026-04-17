// gen_lse.cpp
// LSE atomic microbenchmark generator.

#include "gen_lse.h"
#include "jit_buffer.h"
#include "harness.h"
#include <asmjit/core.h>
#include <asmjit/a64.h>
#include <cstdio>
#include <cstring>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Atomic target ─────────────────────────────────────────────────────────────
//
// All tests operate on this single aligned word. It lives in its own 64-byte
// cache line (no false sharing with adjacent variables). A fresh zero is
// written before each test's JIT function is compiled to ensure a predictable
// start state.
alignas(64) static uint64_t g_atomic_target;

// ── Helpers ───────────────────────────────────────────────────────────────────

static BenchmarkParams make_params(const BenchmarkParams& base,
                                   uint64_t loops, uint32_t unroll) {
    BenchmarkParams p         = base;
    p.loops                   = loops;
    p.instructions_per_loop   = unroll;
    return p;
}

static void run_one(const char* name, JitPool::TestFn fn,
                    const BenchmarkParams& params) {
    if (!fn) { fprintf(stderr, "LSE: skipping %s (compile failed)\n", name); return; }
    benchmark(fn, name, params);
    g_jit_pool->release(fn);
}

// ── Generic atomic loop builder ───────────────────────────────────────────────
//
// Register assignments in the generated code:
//   x19  = outer loop counter (callee-saved; saved/restored in prologue)
//   x20  = address of g_atomic_target (callee-saved; saved/restored)
//   x0   = constant operand: addend for LDADD, swap value for SWP (= 1)
//   x1   = result register: receives the old memory value (discarded)
//
// The emit_body lambda receives the assembler and the current unroll index.
// It must NOT modify x19 or x20.

template<typename F>
static JitPool::TestFn build_atomic_loop(uint64_t loops, uint32_t unroll,
                                          F&& emit_body) {
    g_atomic_target = 0;

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    // ── Prologue ─────────────────────────────────────────────────────────
    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(reinterpret_cast<uint64_t>(&g_atomic_target)));
    a.mov(x0,  Imm(1));        // constant addend / swap value
    a.mov(x1,  Imm(0));        // result register pre-cleared

    // ── Loop (64-byte aligned) ────────────────────────────────────────────
    a.align(AlignMode::kCode, 64);
    Label top = a.new_label();
    a.bind(top);

    for (uint32_t u = 0; u < unroll; ++u)
        emit_body(a, u);

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, top);

    // ── Epilogue ──────────────────────────────────────────────────────────
    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) fprintf(stderr, "build_atomic_loop: compile failed\n");
    return fn;
}

// ── CAS loop builder ─────────────────────────────────────────────────────────
//
// CAS has different register semantics from LDADD/SWP:
//   x0   = expected value (compared against memory; updated to old_mem after)
//   x1   = new value (written to memory when CAS succeeds; held at 0)
//
// We keep mem[x20] = 0, expected = x0 = 0, new = x1 = 0 throughout.
// CAS always succeeds (expected == mem == 0) and writes 0 back, leaving
// both mem and x0 at 0. The data dependency through x0 (each CAS writes the
// old memory value into x0, which the next CAS reads as expected) forces
// serial execution even for the no-ordering CAS variant.
//
// Note: gen_pitfalls.cpp also measures CAS/CASAL in a spinlock-failure context.
// This module measures the raw per-instruction round-trip latency.

static JitPool::TestFn build_cas_loop(uint64_t loops, uint32_t unroll,
                                       bool acquire_release) {
    g_atomic_target = 0;

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(reinterpret_cast<uint64_t>(&g_atomic_target)));
    a.mov(x0,  Imm(0));   // expected = 0 (= current memory value)
    a.mov(x1,  Imm(0));   // new value = 0 (CAS writes 0, no change to mem)

    a.align(AlignMode::kCode, 64);
    Label top = a.new_label();
    a.bind(top);

    for (uint32_t u = 0; u < unroll; ++u) {
        if (acquire_release)
            a.casal(x0, x1, ptr(x20));   // full seq-cst CAS
        else
            a.cas(x0, x1, ptr(x20));     // relaxed CAS
        // After: x0 = old_mem = 0 (data dep chain); mem = 0 (unchanged)
    }

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, top);

    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) fprintf(stderr, "build_cas_loop: compile failed\n");
    return fn;
}

// ── LL/SC loop builder ────────────────────────────────────────────────────────
//
// Each unrolled body: LDAXR x1, [x20] → ADD x1, x1, x0 → STLXR w2, x1, [x20]
//
// No retry branch is emitted. On macOS, the OS scheduler can clear the
// exclusive monitor during a preemption window, causing STLXR to fail (w2≠0).
// A retry loop would livelock indefinitely in that case. Instead we proceed
// regardless; a failed store appears as noise in the CoV. For timing purposes
// the instruction we're measuring is the LDAXR+STLXR pair — a failed store
// still exercises the full exclusive-monitor roundtrip cost.
//
// instructions_per_loop = unroll (each LDAXR+ADD+STLXR triple counted as
// one "operation") so clk/op is directly comparable to the LSE results.

static JitPool::TestFn build_llsc_loop(uint64_t loops, uint32_t unroll) {
    g_atomic_target = 0;

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(16));
    a.stp(x19, x20, ptr(sp));

    a.mov(x19, Imm(loops));
    a.mov(x20, Imm(reinterpret_cast<uint64_t>(&g_atomic_target)));
    a.mov(x0,  Imm(1));   // addend

    a.align(AlignMode::kCode, 64);
    Label top = a.new_label();
    a.bind(top);

    for (uint32_t u = 0; u < unroll; ++u) {
        a.ldaxr(x1, ptr(x20));      // load-acquire-exclusive
        a.add(x1, x1, x0);          // x1 += 1
        a.stlxr(w2, x1, ptr(x20)); // store-release-exclusive (result in w2)
        // No CBNZ retry: see comment above.
    }

    a.sub(x19, x19, Imm(1));
    a.cbnz(x19, top);

    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(16));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) fprintf(stderr, "build_llsc_loop: compile failed\n");
    return fn;
}

// ════════════════════════════════════════════════════════════════════════════
// Test suites
// ════════════════════════════════════════════════════════════════════════════

static void run_lse_atomic_tests(const BenchmarkParams& base,
                                  uint64_t loops, uint32_t unroll) {
    char name[96];

    // ── LDADD: load-and-add ────────────────────────────────────────────────
    //
    // LDADD <Xs>, <Xt>, [<Xn>]:  mem[Xn] += Xs;  Xt = old_mem
    //
    // Four ordering variants reflect the acquire (A) and release (L) suffixes.
    // These directly map to C++ std::atomic memory_order values:
    //   no suffix → relaxed (consume/relaxed)
    //   A suffix  → acquire  (memory_order_acquire)
    //   L suffix  → release  (memory_order_release)
    //   AL suffix → seq_cst  (memory_order_seq_cst, the default)
    //
    // x0 = addend = 1 (constant; never written by LDADD).
    // x1 = result  = old_mem value (discarded; no data dependency chain).
    // Serialisation comes entirely from memory ordering requirements.

    struct {
        const char* label;
        void (*emit)(a64::Assembler&, uint32_t);
    } ldadd_tests[] = {
        { "LDADD   (no ord) ",
          [](a64::Assembler& a, uint32_t) { a.ldadd  (x0, x1, ptr(x20)); } },
        { "LDADDA  (acq)    ",
          [](a64::Assembler& a, uint32_t) { a.ldadda (x0, x1, ptr(x20)); } },
        { "LDADDL  (rel)    ",
          [](a64::Assembler& a, uint32_t) { a.ldaddl (x0, x1, ptr(x20)); } },
        { "LDADDAL (acq+rel)",
          [](a64::Assembler& a, uint32_t) { a.ldaddal(x0, x1, ptr(x20)); } },
    };

    for (const auto& t : ldadd_tests) {
        g_atomic_target = 0;
        auto fn = build_atomic_loop(loops, unroll, t.emit);
        snprintf(name, sizeof(name), "%s (%ux)", t.label, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── STADD: store-add (no return value) ─────────────────────────────────
    //
    // STADD <Xs>, [<Xn>]:  mem[Xn] += Xs  (no old value returned)
    //
    // STADD encodes as LDADD with Xt = XZR. The processor may be able to
    // skip the forwarding path to a result register when Xt=XZR, potentially
    // reducing latency relative to LDADD. On Apple M-series, STADD and LDADD
    // are reported to have identical latency (the result path is not on the
    // critical loop).
    //
    // Only no-ordering and release variants exist; acquire implies a load
    // side-effect which would require a result register.

    struct {
        const char* label;
        void (*emit)(a64::Assembler&, uint32_t);
    } stadd_tests[] = {
        { "STADD   (no ord) ",
          [](a64::Assembler& a, uint32_t) { a.stadd (x0, ptr(x20)); } },
        { "STADDL  (rel)    ",
          [](a64::Assembler& a, uint32_t) { a.staddl(x0, ptr(x20)); } },
    };

    for (const auto& t : stadd_tests) {
        g_atomic_target = 0;
        auto fn = build_atomic_loop(loops, unroll, t.emit);
        snprintf(name, sizeof(name), "%s (%ux)", t.label, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── SWP: atomic exchange ────────────────────────────────────────────────
    //
    // SWP <Xs>, <Xt>, [<Xn>]:  tmp = mem[Xn]; mem[Xn] = Xs; Xt = tmp
    //
    // Useful for test-and-set spinlock: write 1, read 0 → lock acquired.
    // All four ordering variants are provided in ARMv8.1.
    //
    // x0 = swap value = 1 (constant; never written by SWP).
    // x1 = result = old_mem (discarded; SWP always writes the same value 1,
    // so after enough iterations the memory value stabilises at 1 and all
    // subsequent SWPs read 1 as the old value — still a valid throughput
    // measurement of the RMW itself).

    struct {
        const char* label;
        void (*emit)(a64::Assembler&, uint32_t);
    } swp_tests[] = {
        { "SWP     (no ord) ",
          [](a64::Assembler& a, uint32_t) { a.swp  (x0, x1, ptr(x20)); } },
        { "SWPA    (acq)    ",
          [](a64::Assembler& a, uint32_t) { a.swpa (x0, x1, ptr(x20)); } },
        { "SWPL    (rel)    ",
          [](a64::Assembler& a, uint32_t) { a.swpl (x0, x1, ptr(x20)); } },
        { "SWPAL   (acq+rel)",
          [](a64::Assembler& a, uint32_t) { a.swpal(x0, x1, ptr(x20)); } },
    };

    for (const auto& t : swp_tests) {
        g_atomic_target = 0;
        auto fn = build_atomic_loop(loops, unroll, t.emit);
        snprintf(name, sizeof(name), "%s (%ux)", t.label, unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── CAS: compare-and-swap ───────────────────────────────────────────────
    //
    // CAS <Xs>, <Xt>, [<Xn>]:
    //   if mem[Xn] == Xs: mem[Xn] = Xt
    //   Xs = old_mem  (written regardless of comparison outcome)
    //
    // Setup: mem[x20] = 0, x0 (expected) = 0, x1 (new) = 0.
    // Every CAS succeeds (mem always 0, expected always 0) and writes 0 back.
    // Data dependency: x0 = old_mem = 0 after each CAS, but the CPU must
    // wait for the previous CAS result before knowing x0 is 0 (RAW hazard).
    //
    // Note: gen_pitfalls.cpp also benchmarks CAS/CASAL in a spinlock-specific
    // context with a different warm-up and loop structure.

    for (bool acqrel : {false, true}) {
        g_atomic_target = 0;
        auto fn = build_cas_loop(loops, unroll, acqrel);
        snprintf(name, sizeof(name), "%-22s (%ux)",
                 acqrel ? "CASAL   (acq+rel)" : "CAS     (no ord) ", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }

    // ── LDAXR + ADD + STLXR (LL/SC, full ordering) ─────────────────────────
    //
    // The pre-LSE ARMv8.0 exclusive-monitor based RMW. Comparing with LDADDAL
    // reveals whether the hardware fuses LDADDAL into a single micro-op or
    // expands it to an LL/SC sequence internally.
    //
    // On Apple M-series, LDADDAL ≈ LDAXR+STLXR in latency, consistent with
    // hardware fusion. On Cortex-A76 / older cores, LL/SC may be faster than
    // LDADDAL because LDADDAL goes through a "snoop filter" barrier that the
    // LL/SC path avoids.
    //
    // instructions_per_loop = unroll: each LDAXR+ADD+STLXR triple is counted
    // as one "operation" so clk/op is directly comparable to LSE latencies.

    {
        g_atomic_target = 0;
        auto fn = build_llsc_loop(loops, unroll);
        snprintf(name, sizeof(name), "LDAXR+STLXR (acq+rel) (%ux)", unroll);
        run_one(name, fn, make_params(base, loops, unroll));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Public entry point
// ════════════════════════════════════════════════════════════════════════════

void run_lse_tests(const BenchmarkParams& base_params) {
    // Reduce unroll: each atomic is 4–20 cycles; 8 unrolls → ~96% useful time.
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = (base_params.instructions_per_loop > 8)
                          ? 8u : base_params.instructions_per_loop;

    printf("\n── LSE atomics (single-threaded, L1-hot) ──────────────────────\n");
    printf("  clk/op = round-trip latency per atomic RMW on a hot cache line.\n"
           "  All ops target one 64-byte-aligned L1-resident word.\n\n");
    run_lse_atomic_tests(base_params, loops, unroll);
}

} // namespace arm64bench::gen
