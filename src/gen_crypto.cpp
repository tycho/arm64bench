// gen_crypto.cpp
// ARMv8-A cryptography extension microbenchmarks (AES, SHA-256, PMULL, CRC32).
// Split out of gen_fp_simd.cpp; runs as part of --simd.

#include "gen_crypto.h"
#include "gen_common.h"
#include <cstdio>

namespace arm64bench::gen {

using namespace asmjit;
using namespace asmjit::a64;

// ════════════════════════════════════════════════════════════════════════════
// Section 8: Cryptography extensions (ARMv8-A)
// ════════════════════════════════════════════════════════════════════════════
//
// ARMv8-A Cryptography extensions provide single-instruction acceleration for:
//   AES:    AESE/AESD (round), AESMC/AESIMC (MixColumns)
//   SHA-256: SHA256H, SHA256H2 (compression), SHA256SU0/SU1 (message schedule)
//   Poly multiply: PMULL/PMULL2 (poly8×8→16 lanes, or poly64×64→128 for GCM)
//   CRC32:  CRC32B/H/W/X, CRC32CB/CH/CW/CX (Castagnoli variant)
//
// All Apple Silicon, Snapdragon X Elite, and ARMv8.1+ Linux targets support
// these extensions. Instructions are JIT-emitted; no compile-time guards needed.
//
// ── AES microarchitecture notes ──────────────────────────────────────────
//
// Typical ARM cores fuse consecutive AESE+AESMC (and AESD+AESIMC) pairs on
// the same register into a single micro-op, reducing the pair to 1 cycle.
// Apple M-series: 2 AES execution units, each can execute 1 fused pair/cycle.
// At 2 independent AES streams, throughput saturates: 1 round pair/cycle total.
//
// ── PMULL poly64 (GCM) ───────────────────────────────────────────────────
//
// GCM (Galois/Counter Mode) authentication uses PMULL Vd.1Q, Vn.1D, Vm.1D
// to compute a 128-bit carry-less multiply. Latency on Apple M-series ~2 cyc.
//
// Chaining: Vd.1Q (128-bit output) → Vd.1D (lower 64 bits as next input).
//
// ── SHA-256 notes ─────────────────────────────────────────────────────────
//
// SHA256H Q0, Q1, V2.4S implements one round of SHA-256 compression.
// Q0 holds the first half of {a,b,c,d,e,f,g,h}; Q1 holds the second half.
// Both are read and written (or written via SHA256H2). Only Q0 is chained.

static void run_crypto_section(const BenchmarkParams& base,
                              uint64_t loops, uint32_t unroll) {
    char name[80];

    // ── AESE latency ──────────────────────────────────────────────────────
    // AESE V0.16B, V1.16B: V0 ← SubBytes(ShiftRows(V0)) XOR V1
    // V1 = constant round key. Chain through V0.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(1).b16(), Imm(0x5A));  // constant round key
                a.movi(vr(0).b16(), Imm(0x01));  // data
            },
            [](a64::Assembler& a, uint32_t) {
                a.aese(vr(0).b16(), vr(1).b16());
            });
        snprintf(name, sizeof(name), "AESE latency          (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── AESMC latency ─────────────────────────────────────────────────────
    // AESMC V0.16B, V0.16B: V0 ← MixColumns(V0)
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) { a.movi(vr(0).b16(), Imm(0x01)); },
            [](a64::Assembler& a, uint32_t) {
                a.aesmc(vr(0).b16(), vr(0).b16());
            });
        snprintf(name, sizeof(name), "AESMC latency         (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── AESE+AESMC pair latency ───────────────────────────────────────────
    // One AES-128 encryption round: AESE immediately followed by AESMC on
    // the same register. Hardware may fuse this into 1 micro-op.
    // Emits unroll/2 pairs = unroll instructions.
    {
        const uint32_t u2 = (unroll / 2) * 2;
        auto fn = build_loop(loops, u2,
            [](a64::Assembler& a) {
                a.movi(vr(1).b16(), Imm(0x5A));  // round key
                a.movi(vr(0).b16(), Imm(0x01));  // data
            },
            [](a64::Assembler& a, uint32_t u) {
                if (u & 1) a.aesmc(vr(0).b16(), vr(0).b16());
                else       a.aese (vr(0).b16(), vr(1).b16());
            });
        snprintf(name, sizeof(name), "AESE+AESMC latency    (%ux unroll)", u2);
        run_one(name, fn, params_for(base, loops, u2));
    }

    // ── AESE+AESMC throughput ─────────────────────────────────────────────
    // N independent AES data streams, all using the same constant key V(n).
    // Each stream: AESE Vi.16B, Vkey.16B → AESMC Vi.16B, Vi.16B
    // Reveals number of AES execution units (saturation chain count).
    {
        static const uint32_t kChains[] = { 2, 4, 6 };
        for (uint32_t nc : kChains) {
            const uint32_t key_reg = nc;
            const uint32_t u2 = nc * 2;  // 2 instructions per stream
            auto fn = build_loop(loops, u2,
                [nc, key_reg](a64::Assembler& a) {
                    a.movi(vr(key_reg).b16(), Imm(0x5A));
                    for (uint32_t i = 0; i < nc; ++i)
                        a.movi(vr(i).b16(), Imm(static_cast<uint64_t>(i + 1)));
                },
                [key_reg](a64::Assembler& a, uint32_t u) {
                    const uint32_t chain = u / 2;
                    if (u & 1) a.aesmc(vr(chain).b16(), vr(chain).b16());
                    else       a.aese (vr(chain).b16(), vr(key_reg).b16());
                });
            snprintf(name, sizeof(name), "AESE+AESMC tput (%u streams, %ux)", nc, u2);
            run_one(name, fn, params_for(base, loops, u2));
        }
    }

    // ── PMULL poly64 latency (GCM form: 64×64 → 128) ─────────────────────
    // PMULL V0.1Q, V0.1D, V1.1D
    // Chain: V0.1D (lower 64 bits of V0) → V0.1Q (full 128-bit result).
    // V1 = constant multiplier (analogous to GCM authentication key H).
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(1).b16(), Imm(0x03));   // constant multiplier
                a.movi(vr(0).b16(), Imm(0xAA));   // data
            },
            [](a64::Assembler& a, uint32_t) {
                a.pmull(vr(0).q(), vr(0).d(), vr(1).d());
            });
        snprintf(name, sizeof(name), "PMULL poly64 latency  (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── SHA256SU0 latency ─────────────────────────────────────────────────
    // SHA256SU0 V0.4S, V1.4S — message schedule step 0. Chains through V0.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(1).s4(), Imm(0x5A));
                a.movi(vr(0).s4(), Imm(0x01));
            },
            [](a64::Assembler& a, uint32_t) {
                a.sha256su0(vr(0).s4(), vr(1).s4());
            });
        snprintf(name, sizeof(name), "SHA256SU0 latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── SHA256H latency ───────────────────────────────────────────────────
    // SHA256H Q0, Q1, V2.4S — compression round A. Q0 = f(Q0, Q1, V2.4S).
    // Q1 (second state half) and V2 (message words) held constant.
    // Chain through Q0 (first state half: a, b, c, d).
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.movi(vr(2).s4(),  Imm(0x5A));   // message words W[t..t+3]
                a.movi(vr(1).b16(), Imm(0x03));   // second state half (constant)
                a.movi(vr(0).b16(), Imm(0x01));   // first state half (chains)
            },
            [](a64::Assembler& a, uint32_t) {
                a.sha256h(vr(0).q(), vr(1).q(), vr(2).s4());
            });
        snprintf(name, sizeof(name), "SHA256H latency       (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CRC32B latency ────────────────────────────────────────────────────
    // CRC32B W0, W0, W1 — CRC-32 of byte W1[7:0], accumulated in W0.
    // W0 chains (CRC state). W1 = constant data byte.
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xAB));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32b(w0, w0, w1); });
        snprintf(name, sizeof(name), "CRC32B latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CRC32W latency ────────────────────────────────────────────────────
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xABCD1234));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32w(w0, w0, w1); });
        snprintf(name, sizeof(name), "CRC32W latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }

    // ── CRC32X latency ────────────────────────────────────────────────────
    {
        auto fn = build_loop(loops, unroll,
            [](a64::Assembler& a) {
                a.mov(x1, Imm(0xABCD123456789ABCULL));
                a.mov(x0, Imm(0xFFFFFFFF));
            },
            [](a64::Assembler& a, uint32_t) { a.crc32x(w0, w0, x1); });
        snprintf(name, sizeof(name), "CRC32X latency        (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ── FEAT_SHA3: EOR3 / RAX1 / XAR / BCAX ──────────────────────────────────────
//
// The SHA-3 (Keccak) helpers are general-purpose bit operations in disguise:
// EOR3 is a three-input XOR, BCAX is Vn ^ (Vm & ~Va), RAX1 is Vn ^ ROL(Vm, 1)
// per 64-bit lane, XAR is ROR(Vn ^ Vm, #imm) per lane. Compilers use EOR3
// and BCAX for any code that XORs three things, so their latency and port
// count matter well beyond Keccak.

static void run_sha3_section(const BenchmarkParams& base,
                             uint64_t loops, uint32_t unroll) {
    if (!cpu_has(CpuFeature::SHA3)) {
        skip_feature(CpuFeature::SHA3, "EOR3/BCAX/RAX1/XAR");
        return;
    }

    char name[80];

    auto seed = [](a64::Assembler& a, uint32_t nc) {
        // Chains in vr(0..nc-1); constants in vr(nc), vr(nc+1).
        a.movi(vr(nc    ).b16(), Imm(0x5A));
        a.movi(vr(nc + 1).b16(), Imm(0xC3));
        for (uint32_t i = 0; i < nc; ++i)
            a.movi(vr(i).b16(), Imm(static_cast<uint64_t>(i + 1)));
    };

    struct Op {
        const char* label;
        void (*emit)(a64::Assembler&, uint32_t d, uint32_t c0, uint32_t c1);
    };
    const Op ops[] = {
        { "EOR3 v16b", [](a64::Assembler& a, uint32_t d, uint32_t c0, uint32_t c1) {
              a.eor3(vr(d).b16(), vr(d).b16(), vr(c0).b16(), vr(c1).b16()); } },
        { "BCAX v16b", [](a64::Assembler& a, uint32_t d, uint32_t c0, uint32_t c1) {
              a.bcax(vr(d).b16(), vr(d).b16(), vr(c0).b16(), vr(c1).b16()); } },
        { "RAX1 v2d ", [](a64::Assembler& a, uint32_t d, uint32_t c0, uint32_t) {
              a.rax1(vr(d).d2(), vr(d).d2(), vr(c0).d2()); } },
        { "XAR  v2d ", [](a64::Assembler& a, uint32_t d, uint32_t c0, uint32_t) {
              a.xar(vr(d).d2(), vr(d).d2(), vr(c0).d2(), Imm(13)); } },
    };

    for (const Op& op : ops) {
        auto emit = op.emit;
        {
            auto fn = build_loop(loops, unroll,
                [seed](a64::Assembler& a) { seed(a, 1); },
                [emit](a64::Assembler& a, uint32_t) { emit(a, 0, 1, 2); });
            snprintf(name, sizeof(name), "%s latency      (%ux unroll)", op.label, unroll);
            run_one(name, fn, params_for(base, loops, unroll));
        }
        snprintf(name, sizeof(name), "%s tput", op.label);
        chain_sweep(base, loops, unroll, name, { 2, 3, 4, 6, 8, 12, 16 },
            [seed](a64::Assembler& a, uint32_t nc) { seed(a, nc); },
            [emit](a64::Assembler& a, uint32_t nc, uint32_t u) { emit(a, u % nc, nc, nc + 1); });
    }
}

// ── FEAT_SHA512 ──────────────────────────────────────────────────────────────
//
// SHA512H  Qd, Qn, Vm.2D   — hash update, Qd accumulates (chains)
// SHA512SU0 Vd.2D, Vn.2D   — message schedule part 1, Vd accumulates
// SHA512SU1 Vd.2D, Vn.2D, Vm.2D — message schedule part 2, Vd accumulates

static void run_sha512_section(const BenchmarkParams& base,
                               uint64_t loops, uint32_t unroll) {
    if (!cpu_has(CpuFeature::SHA512)) {
        skip_feature(CpuFeature::SHA512, "SHA512H/SHA512SU0/SHA512SU1");
        return;
    }

    char name[80];
    auto seed = [](a64::Assembler& a) {
        a.movi(vr(0).b16(), Imm(0x11));
        a.movi(vr(1).b16(), Imm(0x22));
        a.movi(vr(2).b16(), Imm(0x33));
    };

    {
        auto fn = build_loop(loops, unroll, seed,
            [](a64::Assembler& a, uint32_t) { a.sha512h(vr(0).q(), vr(1).q(), vr(2).d2()); });
        snprintf(name, sizeof(name), "SHA512H latency       (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_loop(loops, unroll, seed,
            [](a64::Assembler& a, uint32_t) { a.sha512h2(vr(0).q(), vr(1).q(), vr(2).d2()); });
        snprintf(name, sizeof(name), "SHA512H2 latency      (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_loop(loops, unroll, seed,
            [](a64::Assembler& a, uint32_t) { a.sha512su0(vr(0).d2(), vr(1).d2()); });
        snprintf(name, sizeof(name), "SHA512SU0 latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
    {
        auto fn = build_loop(loops, unroll, seed,
            [](a64::Assembler& a, uint32_t) { a.sha512su1(vr(0).d2(), vr(1).d2(), vr(2).d2()); });
        snprintf(name, sizeof(name), "SHA512SU1 latency     (%ux unroll)", unroll);
        run_one(name, fn, params_for(base, loops, unroll));
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

void run_crypto_tests(const BenchmarkParams& base_params) {
    const uint64_t loops  = base_params.loops;
    const uint32_t unroll = base_params.instructions_per_loop;
    run_crypto_section(base_params, loops, unroll);
    section("FEAT_SHA3 (EOR3 / BCAX / RAX1 / XAR)");
    run_sha3_section(base_params, loops, unroll);
    section("FEAT_SHA512");
    run_sha512_section(base_params, loops, unroll);
}

} // namespace arm64bench::gen
