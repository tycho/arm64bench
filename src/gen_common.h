#pragma once
// gen_common.h
// Shared scaffolding for the test generators: register tables, the JIT loop
// builder, benchmark-run helpers, throughput chain sweeps, section headers,
// and page-granular memory helpers.
//
// Every generator used to carry its own copy of these. The copies drifted
// (different stack frames, different unroll rounding, different handling of
// a failed compile), so new sections should build on this header instead of
// copying a neighbour.
//
// ── Register conventions inside a build_loop() body ──────────────────────────
//
//   x0–x15   scratch, caller-saved, never touched by the builder
//   x19      loop counter — DO NOT TOUCH
//   x20–x22  saved and restored by the builder; free for the generator's
//            constants and base addresses (set them in `setup`)
//   x30      saved and restored; bodies may BL
//   sp       if scratch_bytes > 0, points at a 16-byte-aligned scratch
//            area of at least that size (typically `mov x9, sp` in setup)
//   v0–v7, v16–v31   caller-saved vector registers, free
//   v8–v15   callee-saved (low 64 bits); NOT saved by the builder — avoid

#include "harness.h"
#include "jit_buffer.h"
#include "cpu_features.h"

#include <asmjit/core.h>
#include <asmjit/a64.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace arm64bench::gen {

// ── Register tables ───────────────────────────────────────────────────────────

// x0..x15 / w0..w15 by index. Out-of-range indices are a programming error.
const asmjit::a64::Gp& xr(uint32_t i);
const asmjit::a64::Gp& wr(uint32_t i);

// ── Benchmark-run helpers ─────────────────────────────────────────────────────

// BenchmarkParams for a function that runs `loops` iterations of
// `insns_per_loop` instructions (bytes_per_insn only for bandwidth tests).
BenchmarkParams params_for(const BenchmarkParams& base, uint64_t loops,
                           uint32_t insns_per_loop, uint32_t bytes_per_insn = 0);

// benchmark() then release. A null fn (compile failure) prints a skip line
// so the gap in the output is explained rather than silent.
void run_one(const char* name, JitPool::TestFn fn, const BenchmarkParams& p);

// Prints "\n── <title> ───…──\n" padded to a fixed width.
void section(const char* title);

// Prints "  (<FEAT_X> not available on this CPU — skipping <what>)".
void skip_feature(CpuFeature f, const char* what);

// ── JIT loop builder ──────────────────────────────────────────────────────────
//
// Emits:
//     sub  sp, sp, #48 ; stp x19,x20 ; stp x21,x22 ; str x30   (frame)
//     [sub sp, sp, #scratch]                                    (optional)
//     mov  x19, #loops
//     setup(a)
//     .align 64
//   top:
//     body(a, u)        for u in [0, unroll)
//     sub  x19, x19, #1 ; cbnz x19, top
//     [add sp, sp, #scratch] ; restore ; ret
//
// setup(a64::Assembler&)            runs once before the loop.
// body (a64::Assembler&, uint32_t)  is called `unroll` times with u = 0..unroll-1.
//
// Returns nullptr (after printing to stderr) if AsmJit rejects the code.

inline constexpr uint32_t kLoopFrameBytes = 48;

inline constexpr auto no_setup = [](asmjit::a64::Assembler&) {};

template<class FSetup, class FBody>
JitPool::TestFn build_loop(uint64_t loops, uint32_t unroll,
                           FSetup&& setup, FBody&& body,
                           uint32_t scratch_bytes = 0)
{
    using namespace asmjit;
    using namespace asmjit::a64;

    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    a.sub(sp, sp, Imm(kLoopFrameBytes));
    a.stp(x19, x20, ptr(sp));
    a.stp(x21, x22, ptr(sp, 16));
    a.str(x30, ptr(sp, 32));

    const uint32_t scratch = (scratch_bytes + 15u) & ~15u;
    if (scratch) a.sub(sp, sp, Imm(scratch));

    a.mov(x19, Imm(loops));
    setup(a);

    a.align(AlignMode::kCode, 64);
    Label top = a.new_label();
    a.bind(top);

    for (uint32_t u = 0; u < unroll; ++u)
        body(a, u);

    a.sub(x19, x19, Imm(1));    // SUB, not SUBS: leaves NZCV alone
    a.cbnz(x19, top);

    if (scratch) a.add(sp, sp, Imm(scratch));
    a.ldr(x30, ptr(sp, 32));
    a.ldp(x21, x22, ptr(sp, 16));
    a.ldp(x19, x20, ptr(sp));
    a.add(sp, sp, Imm(kLoopFrameBytes));
    a.ret(x30);

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) fprintf(stderr, "build_loop: JIT compile failed\n");
    return fn;
}

// ── Throughput chain sweeps ───────────────────────────────────────────────────
//
// A chain sweep runs the same one-instruction body over nc independent
// dependency chains (registers 0..nc-1) for each nc in `chains`, and names
// the results "<prefix> (<nc> chains, <unroll>x unroll)". The saturation
// point reveals the number of execution units for that instruction class.

inline constexpr uint32_t kDefaultChains[] = { 2, 3, 4, 6, 8 };
inline constexpr uint32_t kWideChains[]    = { 2, 3, 4, 6, 8, 10, 12, 16 };

// Unroll rounded down to a multiple of nc × group so every chain gets the
// same number of instructions per iteration. Never returns 0: if the unroll
// is smaller than nc × group, that product is used instead.
uint32_t chain_unroll(uint32_t unroll, uint32_t nc, uint32_t group = 1);

// setup(a, nc)     seeds the nc chain registers (and any constants).
// body (a, nc, u)  emits exactly ONE instruction for chain (u % nc).
// group            > 1 when consecutive u values form a fixed pattern
//                  (e.g. alternating ADDS/CSEL) that must not be split
//                  across chains — unroll is rounded to nc × group.
template<class FSetup, class FBody>
void chain_sweep(const BenchmarkParams& base, uint64_t loops, uint32_t unroll,
                 const char* name_prefix, std::span<const uint32_t> chains,
                 FSetup&& setup, FBody&& body, uint32_t group = 1)
{
    char name[96];
    for (const uint32_t nc : chains) {
        const uint32_t au = chain_unroll(unroll, nc, group);
        auto fn = build_loop(loops, au,
            [&](asmjit::a64::Assembler& a)             { setup(a, nc); },
            [&](asmjit::a64::Assembler& a, uint32_t u) { body(a, nc, u); });
        snprintf(name, sizeof(name), "%s (%u chains, %ux unroll)",
                 name_prefix, nc, au);
        run_one(name, fn, params_for(base, loops, au));
    }
}

// ── Page-granular memory helpers ──────────────────────────────────────────────

// Anonymous, zero-filled, page-aligned mapping (mmap / VirtualAlloc).
// Returns nullptr on failure.
void* alloc_pages(size_t bytes);
void  free_pages(void* p, size_t bytes);

// Touch one byte per 4 KB page so the pages are resident before timing.
void commit_pages(void* p, size_t bytes);

// xorshift64 with the fixed seed every pointer chain in arm64bench uses,
// so chains are identical run to run (TLB pressure depends on the order).
inline constexpr uint64_t kChaseSeed = 0xDEADBEEF12345678ULL;
uint64_t xorshift64(uint64_t& state);

// Random cyclic pointer chain over buf[0..size): one node every `stride`
// bytes, and each node's next-pointer stored at (node + offset) — offset
// lets the misalignment tests place the pointer across word or line
// boundaries. Visits every node once before repeating. Returns the head
// (the address to load from first), or nullptr if there are fewer than
// two nodes or the index array cannot be allocated.
void* build_pointer_ring(void* buf, size_t size, size_t stride, size_t offset = 0);

} // namespace arm64bench::gen
