// gen_c2c.cpp
// Core-to-core transfer latency and contended atomics. See gen_c2c.h.
//
// ── Protocol state ───────────────────────────────────────────────────────────
//
// Every shared word sits alone in its own 64-byte line. The round-trip
// counters are never reset: the timed function loads the current value in
// its setup and continues from there, so a warm-up call, a reference call
// and a timed call all see a consistent protocol whatever ran before them.
//
// The responder checks the stop line only while idle (no request pending),
// which is where it sits once the main thread's last request is answered.

#include "gen_c2c.h"
#include "gen_common.h"
#include "affinity.h"

#include <atomic>
#include <cstdio>
#include <thread>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ── Shared lines ──────────────────────────────────────────────────────────────

struct alignas(64) Line {
    std::atomic<uint64_t> v{0};
    char pad[56];
};

static Line g_flag;      // 1-line protocol
static Line g_req;       // 2-line protocol, written by the main thread
static Line g_ack;       // 2-line protocol, written by the partner
static Line g_counter;   // contended atomic target
static Line g_stop;      // partner exits when non-zero
static Line g_ready;     // partner sets when placed and running

static uint64_t addr(const Line& l) { return reinterpret_cast<uint64_t>(&l.v); }

// ── Responders (run by the partner thread) ────────────────────────────────────

enum class Probe { OneLineOrdered, OneLinePlain, TwoLine, ContendedAdd };

static const char* probe_name(Probe p) {
    switch (p) {
        case Probe::OneLineOrdered: return "1-line round trip, LDAR/STLR";
        case Probe::OneLinePlain:   return "1-line round trip, LDR/STR";
        case Probe::TwoLine:        return "2-line round trip, LDAR/STLR";
        case Probe::ContendedAdd:   return "LDADDAL, contended";
    }
    return "";
}

// A responder is a plain function with no frame: it uses only x0–x11 and
// returns when the stop line is set.
static JitPool::TestFn build_responder(Probe p) {
    CodeHolder code;
    g_jit_pool->init_code_holder(code);
    a64::Assembler a(&code);

    Label top = a.new_label(), respond = a.new_label();
    a.mov(x11, Imm(addr(g_stop)));

    switch (p) {
    case Probe::OneLineOrdered:
    case Probe::OneLinePlain: {
        const bool ord = (p == Probe::OneLineOrdered);
        a.mov(x9, Imm(addr(g_flag)));
        a.bind(top);
        if (ord) a.ldar(x0, ptr(x9)); else a.ldr(x0, ptr(x9));
        a.tbnz(x0, Imm(0), respond);          // odd = request pending
        a.ldr(x1, ptr(x11));
        a.cbz(x1, top);
        a.ret(x30);
        a.bind(respond);
        a.add(x0, x0, Imm(1));
        if (ord) a.stlr(x0, ptr(x9)); else a.str(x0, ptr(x9));
        a.b(top);
        break;
    }
    case Probe::TwoLine:
        a.mov(x9,  Imm(addr(g_req)));
        a.mov(x10, Imm(addr(g_ack)));
        a.ldr(x2, ptr(x10));                  // last value acknowledged
        a.bind(top);
        a.ldar(x0, ptr(x9));
        a.cmp(x0, x2);
        a.b(CondCode::kNE, respond);
        a.ldr(x1, ptr(x11));
        a.cbz(x1, top);
        a.ret(x30);
        a.bind(respond);
        a.mov(x2, x0);
        a.stlr(x0, ptr(x10));
        a.b(top);
        break;
    case Probe::ContendedAdd:
        a.mov(x9, Imm(addr(g_counter)));
        a.mov(x0, Imm(1));
        a.bind(top);
        a.ldaddal(x0, x1, ptr(x9));
        a.ldr(x2, ptr(x11));
        a.cbz(x2, top);
        a.ret(x30);
        break;
    }

    JitPool::TestFn fn = g_jit_pool->compile(code);
    if (!fn) fprintf(stderr, "build_responder: JIT compile failed\n");
    return fn;
}

// ── Timed functions (run by the main thread) ──────────────────────────────────

static constexpr uint64_t kRoundTrips = 100'000;   // ~100 ns each: 10 ms per call
static constexpr uint64_t kAddLoops   = 25'000;
static constexpr uint32_t kAddUnroll  = 8;

static JitPool::TestFn build_probe(Probe p, uint64_t loops) {
    switch (p) {
    case Probe::OneLineOrdered:
    case Probe::OneLinePlain: {
        const bool ord = (p == Probe::OneLineOrdered);
        return build_loop(loops, 1,
            [](a64::Assembler& a) {
                a.mov(x20, Imm(addr(g_flag)));
                a.ldr(x0, ptr(x20));           // current (even) value
            },
            [ord](a64::Assembler& a, uint32_t) {
                Label spin = a.new_label();
                a.add(x0, x0, Imm(1));         // odd: request
                if (ord) a.stlr(x0, ptr(x20)); else a.str(x0, ptr(x20));
                a.add(x0, x0, Imm(1));         // even: the expected reply
                a.bind(spin);
                if (ord) a.ldar(x2, ptr(x20)); else a.ldr(x2, ptr(x20));
                a.cmp(x2, x0);
                a.b(CondCode::kNE, spin);
            });
    }
    case Probe::TwoLine:
        return build_loop(loops, 1,
            [](a64::Assembler& a) {
                a.mov(x20, Imm(addr(g_req)));
                a.mov(x21, Imm(addr(g_ack)));
                a.ldr(x0, ptr(x20));
            },
            [](a64::Assembler& a, uint32_t) {
                Label spin = a.new_label();
                a.add(x0, x0, Imm(1));
                a.stlr(x0, ptr(x20));
                a.bind(spin);
                a.ldar(x2, ptr(x21));
                a.cmp(x2, x0);
                a.b(CondCode::kNE, spin);
            });
    case Probe::ContendedAdd:
        return build_loop(loops, kAddUnroll,
            [](a64::Assembler& a) {
                a.mov(x20, Imm(addr(g_counter)));
                a.mov(x0, Imm(1));
            },
            [](a64::Assembler& a, uint32_t) { a.ldaddal(x0, x1, ptr(x20)); });
    }
    return nullptr;
}

// ── Partner thread ────────────────────────────────────────────────────────────

struct Placement {
    int  cpu;          // >= 0: pin to this logical CPU
    bool efficiency;   // macOS: efficiency cluster via QoS
};

struct Partner {
    std::thread     thread;
    JitPool::TestFn fn = nullptr;

    bool start(Probe p, Placement where) {
        fn = build_responder(p);
        if (!fn) return false;
        g_stop.v.store(0);
        g_ready.v.store(0);
        thread = std::thread([this, where] {
            if (where.cpu >= 0) pin_thread_to_cpu(static_cast<uint32_t>(where.cpu));
            set_thread_cluster_hint(where.efficiency);
            g_ready.v.store(1);
            fn();
        });
        while (g_ready.v.load() == 0) {}
        return true;
    }

    void stop() {
        g_stop.v.store(1);
        if (thread.joinable()) thread.join();
        if (fn) g_jit_pool->release(fn);
        fn = nullptr;
    }
};

// ── Driver ────────────────────────────────────────────────────────────────────

static void run_probe(const BenchmarkParams& base, Probe p, Placement where,
                      const char* label) {
    const bool add = (p == Probe::ContendedAdd);
    const uint64_t loops  = scale_loops(add ? kAddLoops : kRoundTrips);
    const uint32_t unroll = add ? kAddUnroll : 1;

    Partner partner;
    if (!partner.start(p, where)) return;
    auto fn = build_probe(p, loops);
    char name[96];
    snprintf(name, sizeof(name), "%-28s %s", probe_name(p), label);
    run_one(name, fn, params_for(base, loops, unroll));
    partner.stop();
}

void run_c2c_tests(const BenchmarkParams& base_params) {
    uint32_t cpus[kMaxCpus];
    uint32_t n = allowed_cpus(cpus, kMaxCpus);
    if (n > kMaxCpus) n = kMaxCpus;
    if (n < 2) {
        printf("  (only one CPU available to this process — skipping core-to-core tests)\n");
        return;
    }

    // Uncontended reference for the atomic probe: no partner at all.
    section("Core-to-core: uncontended reference");
    {
        const uint64_t loops = scale_loops(kAddLoops);
        auto fn = build_probe(Probe::ContendedAdd, loops);
        run_one("LDADDAL, no partner", fn, params_for(base_params, loops, kAddUnroll));
    }

    static constexpr Probe kProbes[] = {
        Probe::OneLineOrdered, Probe::OneLinePlain, Probe::TwoLine, Probe::ContendedAdd,
    };
    char label[48], title[96];

    // Home = the first CPU of the set main() pinned this thread to (the chosen
    // cluster), so the matrix is measured from the core type every other test
    // ran on. The partner sweeps every other CPU the process may use.
    uint32_t mine[kMaxCpus];
    uint32_t nm = thread_cpus(mine, kMaxCpus);
    if (nm > kMaxCpus) nm = kMaxCpus;
    const uint32_t home = nm ? mine[0] : cpus[0];

    if (affinity_supported() && pin_thread_to_cpu(home)) {
        for (const Probe p : kProbes) {
            snprintf(title, sizeof(title), "Core-to-core: %s, this thread on cpu %u",
                     probe_name(p), home);
            section(title);
            for (uint32_t i = 0; i < n; ++i) {
                if (cpus[i] == home) continue;
                snprintf(label, sizeof(label), "partner cpu %2u", cpus[i]);
                run_probe(base_params, p, Placement{ static_cast<int>(cpus[i]), false }, label);
            }
        }
        if (nm) pin_thread_to_cpus(mine, nm); else unpin_thread();
    } else {
        section("Core-to-core: no thread affinity on this OS; partner placed by QoS class");
        printf("  'P' = partner at user-interactive QoS (performance cluster),\n"
               "  'E' = partner at background QoS (efficiency cluster). This thread\n"
               "  runs at user-interactive QoS. Exact cores are the scheduler's choice.\n");
        for (const Probe p : kProbes) {
            run_probe(base_params, p, Placement{ -1, false }, "partner P");
            run_probe(base_params, p, Placement{ -1, true  }, "partner E");
        }
    }
}

} // namespace arm64bench::gen
