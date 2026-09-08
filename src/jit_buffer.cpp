// jit_buffer.cpp
// JIT executable memory pool implementation.
//
// This file is intentionally thin. The heavy lifting (W^X memory management,
// platform-specific icache flushing, Apple Silicon MAP_JIT + jit_write_protect)
// is handled entirely by AsmJit's JitRuntime. Our role is:
//   1. Provide a clean ownership interface (compile / release).
//   2. Translate AsmJit error codes into human-readable diagnostics.
//   3. Hold the single process-wide JitRuntime instance.

#include "jit_buffer.h"
#include <cstdio>

namespace arm64bench {

JitPool* g_jit_pool = nullptr;

// Without a handler AsmJit records an error and silently skips the offending
// instruction, so a rejected encoding turns into a function with a hole in
// it. Print every error as it happens; compile() also fails the function.
namespace {
struct PrintingErrorHandler : asmjit::ErrorHandler {
    void handle_error(asmjit::Error err, const char* message,
                      asmjit::BaseEmitter* /*origin*/) override {
        fprintf(stderr, "asmjit error %u: %s\n", static_cast<unsigned>(err), message);
        fflush(stderr);
    }
};
PrintingErrorHandler s_error_handler;
} // namespace

JitPool::JitPool() {
    // JitRuntime's constructor detects the current architecture and OS and
    // configures the appropriate memory allocation strategy:
    //   macOS/iOS AArch64:  mmap(MAP_JIT) + pthread_jit_write_protect_np
    //   Windows AArch64:    VirtualAlloc + VirtualProtect + FlushInstructionCache
    //   Linux AArch64:      mmap + mprotect + __builtin___clear_cache
    //
    // Nothing extra to do here; the default constructor handles it all.
}

void JitPool::init_code_holder(asmjit::CodeHolder& code) {
    const asmjit::Error err = code.init(_rt.environment(), _rt.cpu_features());
    if (err != asmjit::kErrorOk) {
        fprintf(stderr,
                "JitPool::init_code_holder failed: %s\n",
                asmjit::stringify_error(err));
    }
    code.set_error_handler(&s_error_handler);
}

JitPool::TestFn JitPool::compile(asmjit::CodeHolder& code) {
    TestFn fn = nullptr;
    const asmjit::Error err = _rt.add(&fn, &code);
    if (err != asmjit::kErrorOk) {
        fprintf(stderr,
                "JitPool::compile failed: %s\n",
                asmjit::stringify_error(err));
        return nullptr;
    }
    return fn;
}

void JitPool::release(TestFn fn) {
    if (!fn)
        return;
    const asmjit::Error err = _rt.release(fn);
    if (err != asmjit::kErrorOk) {
        fprintf(stderr,
                "JitPool::release failed: %s\n",
                asmjit::stringify_error(err));
    }
}

} // namespace arm64bench
