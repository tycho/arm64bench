cmake_minimum_required(VERSION 3.25)

project(arm64bench LANGUAGES CXX)

# ── Toolchain requirements ─────────────────────────────────────────────────────
# This project targets Clang on macOS (Apple Silicon), Windows ARM64, and Linux
# AArch64. MSVC is not supported; clang-cl on Windows is acceptable.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "arm64bench requires Clang. Use -T ClangCL or the LLVM toolchain.")
endif()
if(MSVC)
	# Use multi-processor compilation in MSVC for faster builds
	add_compile_options(/MP)
endif()

# AsmJit exports cxx_std_20 as a PUBLIC compile feature, so every target that
# links it is compiled as C++20 regardless of this setting. State it plainly.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── AsmJit dependency ──────────────────────────────────────────────────────────
# AsmJit is used as the JIT code generation backend. We build it as a static
# library so there are no shared library deployment concerns.
#

# Must be set before FetchContent_MakeAvailable so AsmJit's CMakeLists sees it.
set(ASMJIT_STATIC ON CACHE BOOL "Build AsmJit as a static library" FORCE)

# Suppress AsmJit's own install rules to keep our install target clean.
set(ASMJIT_NO_INSTALL ON CACHE BOOL "" FORCE)

add_subdirectory(contrib/asmjit)

# ── Sources ────────────────────────────────────────────────────────────────────
# Everything except main() lives in a static library so that the benchmark
# executable and the self-test executable share one build of the harness,
# timer, cycle counter, JIT pool, and test generators.
add_library(arm64bench_core STATIC
    src/timer.cpp
    src/harness.cpp
    src/cycle_counter.cpp
    src/cpu_features.cpp
    src/jit_buffer.cpp
    src/calibrate.cpp
    src/gen_common.cpp
    src/gen_integer.cpp
    src/gen_memory.cpp
    src/gen_branch.cpp
    src/gen_fp_simd.cpp
    src/gen_crypto.cpp
    src/gen_i8mm.cpp
    src/gen_bf16.cpp
    src/gen_lse.cpp
    src/gen_pitfalls.cpp
    src/gen_ooo.cpp
    src/gen_sve.cpp
    src/gen_mlp.cpp
    src/gen_frontend.cpp
    src/gen_icache.cpp
    src/gen_prefetch.cpp
)
target_include_directories(arm64bench_core PUBLIC src)
target_link_libraries(arm64bench_core PUBLIC asmjit::asmjit)

add_executable(arm64bench src/main.cpp)
target_link_libraries(arm64bench PRIVATE arm64bench_core)

# ── Compile options (our targets only, not AsmJit) ────────────────────────────
# AsmJit itself does not use C++ exceptions or RTTI so it compiles cleanly
# under these flags, but we only apply them to our own targets to be safe.
function(arm64bench_configure_target tgt)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND MSVC)
        # clang-cl: MSVC-compatible flag spelling
        target_compile_options(${tgt} PRIVATE
            /EHs-c-         # disable C++ exceptions
            /GR-            # disable RTTI
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Z7>
        )
    else()
        # Clang on macOS / Linux
        target_compile_options(${tgt} PRIVATE
            -fno-exceptions
            -fno-rtti
            -Wall
            -Wextra
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        )
    endif()

    # ── Platform-specific ──────────────────────────────────────────────────
    if(APPLE)
        target_compile_definitions(${tgt} PRIVATE BENCH_PLATFORM_MACOS=1)

        # For signed distribution builds, the executable needs the
        # com.apple.security.cs.allow-jit entitlement for MAP_JIT.
        # Unsigned local development builds work without it.
        # See: https://developer.apple.com/documentation/bundleresources/entitlements

    elseif(WIN32)
        target_compile_definitions(${tgt} PRIVATE
            BENCH_PLATFORM_WINDOWS=1
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            UNICODE
        )
        # Link against the performance counter API (included in kernel32, but be explicit)
        target_link_libraries(${tgt} PRIVATE kernel32)

    else()
        target_compile_definitions(${tgt} PRIVATE BENCH_PLATFORM_LINUX=1)
        # perf_event_open lives in the kernel; link pthreads for scheduling control
        target_link_libraries(${tgt} PRIVATE pthread)
    endif()
endfunction()

arm64bench_configure_target(arm64bench_core)
arm64bench_configure_target(arm64bench)

# ── Self-test and CTest registration ──────────────────────────────────────────
# arm64bench_selftest verifies the measurement machinery itself (timer,
# cycle counter, calibration, harness call accounting, and a few results
# that every ARM64 core must reproduce). `arm64bench --all --smoke` executes
# every generated test once. Both are registered with CTest.
option(ARM64BENCH_BUILD_TESTS "Build arm64bench_selftest and register CTest tests" ON)
if(ARM64BENCH_BUILD_TESTS)
    enable_testing()

    add_executable(arm64bench_selftest tests/selftest.cpp)
    target_link_libraries(arm64bench_selftest PRIVATE arm64bench_core)
    arm64bench_configure_target(arm64bench_selftest)

    add_test(NAME selftest COMMAND arm64bench_selftest)
    add_test(NAME smoke    COMMAND arm64bench --all --smoke)
    set_tests_properties(selftest smoke PROPERTIES TIMEOUT 600)
endif()

# ── Default build type ─────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type set; defaulting to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
