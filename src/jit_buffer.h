#pragma once
// jit_buffer.h
// JIT executable memory pool: compiles AsmJit CodeHolders into callable
// function pointers, handling W^X (Write XOR Execute) memory requirements
// on all target platforms.
//
// Platform notes:
//
//   Apple Silicon (macOS, iOS):
//     Hardware enforces W^X strictly. Pages must be mapped with MAP_JIT, and
//     the write-protect state is toggled per-thread via
//     pthread_jit_write_protect_np(). For signed distribution builds, the
//     process needs the com.apple.security.cs.allow-jit entitlement.
//     AsmJit's JitRuntime handles all of this internally.
//
//   Windows ARM64:
//     Uses VirtualAlloc(PAGE_READWRITE) for writing, then VirtualProtect to
//     PAGE_EXECUTE_READ. FlushInstructionCache() is required after writing
//     and before execution, or the CPU may fetch stale icache lines.
//     AsmJit's JitRuntime handles this.
//
//   Linux AArch64:
//     Uses mmap(PROT_READ|PROT_WRITE) → mprotect(PROT_READ|PROT_EXEC).
//     __builtin___clear_cache() is required after writing.
//     AsmJit's JitRuntime handles this.
//
// Ownership model:
//   compile() allocates executable memory and returns a raw function pointer.
//   The caller owns that pointer and must call release() when done.
//   Typical lifetime: generate once per test variant at startup, run
//   thousands of times during benchmarking, release at process exit.

#include <cstddef>
#include <cstdint>
#include <asmjit/core.h>
#include <asmjit/a64.h>

namespace arm64bench {

class JitPool {
public:
    // Signature of JIT-compiled test functions. No arguments, no return value.
    // All parameters (loop count, registers to use, etc.) are baked in at
    // JIT compile time. The harness calls these as plain C function pointers.
    using TestFn = void (*)();

    JitPool();
    ~JitPool() = default;

    JitPool(const JitPool&)            = delete;
    JitPool& operator=(const JitPool&) = delete;

    // Compile the code in `code` into executable memory and return a callable
    // function pointer. Returns nullptr and prints an error if compilation
    // fails (out of memory, encoding error, etc.).
    //
    // The CodeHolder must have been initialized against this pool's runtime
    // environment (use init_code_holder() below, or construct CodeHolder with
    // runtime().environment() explicitly).
    //
    // After this call, `code` is finalized and must not be modified.
    TestFn compile(asmjit::CodeHolder& code);

    // Release a function pointer previously returned by compile().
    // After this call, the pointer is invalid and must not be called.
    void release(TestFn fn);

    // Initialize a CodeHolder for use with this pool.
    // Equivalent to code.init(runtime().environment()), but provided here
    // so callers don't need to reach through to the underlying JitRuntime.
    void init_code_holder(asmjit::CodeHolder& code);

    // Direct access to the underlying AsmJit runtime, for callers that need
    // it (e.g. to attach an error handler or logger during development).
    asmjit::JitRuntime&       runtime()       { return _rt; }
    const asmjit::JitRuntime& runtime() const { return _rt; }

private:
    asmjit::JitRuntime _rt;
};

// Process-wide JIT pool. Created in main() before any test generators run,
// and valid for the entire lifetime of the process.
//
// All test generators should compile their functions through this singleton
// so that the underlying allocator can coalesce allocations efficiently.
extern JitPool* g_jit_pool;

} // namespace arm64bench
