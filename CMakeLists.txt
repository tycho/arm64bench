cmake_minimum_required(VERSION 3.20)
project(arm64bench LANGUAGES CXX)

# ── Toolchain requirements ─────────────────────────────────────────────────────
# This project targets Clang on macOS (Apple Silicon), Windows ARM64, and Linux
# AArch64. MSVC is not supported; clang-cl on Windows is acceptable.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "arm64bench requires Clang. Use -T ClangCL or the LLVM toolchain.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── AsmJit dependency ──────────────────────────────────────────────────────────
# AsmJit is used as the JIT code generation backend. We build it as a static
# library so there are no shared library deployment concerns.
#
# TODO: Pin GIT_TAG to a specific commit hash for reproducible builds.
include(FetchContent)
FetchContent_Declare(
    asmjit
    GIT_REPOSITORY https://github.com/asmjit/asmjit.git
    GIT_TAG        master
)
# Must be set before FetchContent_MakeAvailable so AsmJit's CMakeLists sees it.
set(ASMJIT_STATIC ON CACHE BOOL "Build AsmJit as a static library" FORCE)

# Suppress AsmJit's own install rules to keep our install target clean.
set(ASMJIT_NO_INSTALL ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(asmjit)

# ── Sources ────────────────────────────────────────────────────────────────────
add_executable(arm64bench
    src/main.cpp
    src/timer.cpp
    src/harness.cpp
    src/jit_buffer.cpp
    src/calibrate.cpp
    src/gen_integer.cpp
    src/gen_memory.cpp
    src/gen_branch.cpp
    src/gen_fp_simd.cpp
)

target_include_directories(arm64bench PRIVATE src)
target_link_libraries(arm64bench PRIVATE asmjit::asmjit)

# ── Compile options (our target only, not AsmJit) ─────────────────────────────
# AsmJit itself does not use C++ exceptions or RTTI so it compiles cleanly
# under these flags, but we only apply them to our own target to be safe.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MSVC)
    # clang-cl: MSVC-compatible flag spelling
    target_compile_options(arm64bench PRIVATE
        /EHs-c-         # disable C++ exceptions
        /GR-            # disable RTTI
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Z7>
    )
else()
    # Clang on macOS / Linux
    target_compile_options(arm64bench PRIVATE
        -fno-exceptions
        -fno-rtti
        -Wall
        -Wextra
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    )
endif()

# ── Platform-specific ──────────────────────────────────────────────────────────
if(APPLE)
    target_compile_definitions(arm64bench PRIVATE BENCH_PLATFORM_MACOS=1)

    # For signed distribution builds, the executable needs the
    # com.apple.security.cs.allow-jit entitlement for MAP_JIT.
    # Unsigned local development builds work without it.
    # See: https://developer.apple.com/documentation/bundleresources/entitlements

elseif(WIN32)
    target_compile_definitions(arm64bench PRIVATE
        BENCH_PLATFORM_WINDOWS=1
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        UNICODE
    )
    # Link against the performance counter API (included in kernel32, but be explicit)
    target_link_libraries(arm64bench PRIVATE kernel32)

else()
    target_compile_definitions(arm64bench PRIVATE BENCH_PLATFORM_LINUX=1)
    # perf_event_open lives in the kernel; link pthreads for scheduling control
    target_link_libraries(arm64bench PRIVATE pthread)
endif()

# ── Default build type ─────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type set; defaulting to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
