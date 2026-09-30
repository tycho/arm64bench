// x86_tso_stlf.c
// Store→load chains as x86-64 code, to run under Rosetta on Apple Silicon.
//
// arm64bench's JIT emits arm64 only, and macOS offers no way to put a native
// arm64 thread into Apple's TSO memory model. Rosetta does run its translated
// code in that mode, and it translates a MOV store / MOV load / LEA chain
// almost one to one, so this is the way to see what `--pitfalls --filter stlf`
// would read with TSO on, on the same core and OS:
//
//     clang -O2 -arch x86_64 tools/x86_tso_stlf.c -o x86_tso_stlf && ./x86_tso_stlf
//
// Each link adds 1 to the value so load value prediction cannot break the
// chain; the unit is one LEA (= one cycle) and every row includes that LEA,
// as the stlf "var" rows include their EOR. On M5 (2026-09-30) the rows where
// the load is contained in one store match the native arm64 numbers, and the
// rows where it is not (a load wider than the store, or reading two stores)
// cost 12 clk more, which is exactly what the native arm64 rows read inside a
// Docker Desktop Linux VM. See CLAUDE.md, "Store-to-load forwarding: value
// prediction, wide loads, vector registers".
//
// Not part of the CMake build.

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#if !defined(__x86_64__)
#error "x86-64 only: build with -arch x86_64 and run under Rosetta"
#endif

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

#define R8(x) x x x x x x x x

// rax carries the chain, rsi is the slot, rcx = 1.
#define CHAIN(name, body)                                                     \
    __attribute__((noinline)) static void name(uint64_t n, void* buf) {       \
        uint64_t x = 1, one = 1;                                              \
        for (uint64_t i = 0; i < n; i++)                                      \
            __asm__ volatile(R8(body)                                         \
                             : "+a"(x)                                        \
                             : "S"(buf), "c"(one)                             \
                             : "memory", "cc", "xmm0", "xmm1");               \
    }

#define NEXT "lea 1(%%rax), %%rax\n"

CHAIN(t_lea,       "lea (%%rax,%%rcx), %%rax\n")
CHAIN(t_q_q,       "movq %%rax, (%%rsi)\n movq (%%rsi), %%rax\n" NEXT)
CHAIN(t_l_l,       "movl %%eax, (%%rsi)\n movl (%%rsi), %%eax\n" NEXT)
CHAIN(t_q_l,       "movq %%rax, (%%rsi)\n movl (%%rsi), %%eax\n" NEXT)
CHAIN(t_l_q,       "movl %%eax, (%%rsi)\n movq (%%rsi), %%rax\n" NEXT)
CHAIN(t_b_l,       "movb %%al, (%%rsi)\n movl (%%rsi), %%eax\n" NEXT)
CHAIN(t_movq_rt,   "movq %%rax, %%xmm0\n movq %%xmm0, %%rax\n" NEXT)
CHAIN(t_qq_x,      "movq %%rax, (%%rsi)\n movq %%rax, 8(%%rsi)\n"
                   "movdqa (%%rsi), %%xmm0\n movq %%xmm0, %%rax\n" NEXT)
CHAIN(t_q_x,       "movq %%rax, (%%rsi)\n movdqa (%%rsi), %%xmm0\n"
                   "movq %%xmm0, %%rax\n" NEXT)
CHAIN(t_q_x8,      "movq %%rax, (%%rsi)\n movq (%%rsi), %%xmm0\n"
                   "movq %%xmm0, %%rax\n" NEXT)
CHAIN(t_x_q,       "movq %%rax, %%xmm0\n movdqa %%xmm0, (%%rsi)\n"
                   "movq (%%rsi), %%rax\n" NEXT)
CHAIN(t_x_x,       "movq %%rax, %%xmm0\n movdqa %%xmm0, (%%rsi)\n"
                   "movdqa (%%rsi), %%xmm1\n movq %%xmm1, %%rax\n" NEXT)

static double run(void (*fn)(uint64_t, void*), void* buf) {
    const uint64_t n = 2000000;
    double best = 1e30;
    for (int r = 0; r < 9; r++) {
        const uint64_t t0 = now_ns();
        fn(n, buf);
        const double t = (double)(now_ns() - t0) / ((double)n * 8.0);
        if (t < best) best = t;
    }
    return best;
}

int main(void) {
    static _Alignas(64) uint8_t buf[64];

    run(t_lea, buf);
    const double lea = run(t_lea, buf);
    printf("LEA chain: %.3f ns per link (one cycle)\n\n", lea);

#define REPORT(label, fn) printf("  %-44s %6.2f clk\n", label, run(fn, buf) / lea)
    REPORT("mov q -> mov q                (matched)", t_q_q);
    REPORT("mov l -> mov l                (matched)", t_l_l);
    REPORT("mov q -> mov l                (narrower)", t_q_l);
    REPORT("mov l -> mov q                (WIDER)", t_l_q);
    REPORT("mov b -> mov l                (WIDER)", t_b_l);
    REPORT("movq xmm,r; movq r,xmm        (no memory)", t_movq_rt);
    REPORT("mov q, mov q -> movdqa; movq  (TWO STORES)", t_qq_x);
    REPORT("mov q -> movdqa; movq         (WIDER)", t_q_x);
    REPORT("mov q -> movq xmm; movq       (same size)", t_q_x8);
    REPORT("movq; movdqa -> mov q         (narrower)", t_x_q);
    REPORT("movq; movdqa -> movdqa; movq  (matched)", t_x_x);
    return 0;
}
