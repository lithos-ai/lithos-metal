# The Metal runtime: an Objective-C++ core bound to Python with nanobind as ``monolith.runtime._native``.
# Build:  cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(pwd)/.venv/bin/python && cmake --build build
# The module is written next to monolith/runtime/ so ``import monolith.runtime`` picks it up from the source tree.
cmake_minimum_required(VERSION 3.24)
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
                OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE NB_RC)
if(NOT NB_RC EQUAL 0)
  message(FATAL_ERROR "nanobind is not installed for ${Python_EXECUTABLE} (pip install nanobind)")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_native NB_STATIC src/native.mm src/metal_core.mm src/basalt_cpu.cpp src/basalt_cpu_expf.cpp)
set_source_files_properties(src/native.mm src/metal_core.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
# basalt's CPU kernels transcribe two AVX-512 roundings (a vmulps then a vaddps) that a contracted fmla would
# collapse into one, changing the activation codes. Measured, Apple clang keeps them apart at every contraction
# setting while they are separate statements, so this pins the guarantee rather than repairing today's output; the
# Metal core keeps the default. These files must never see -ffast-math, which is a different hazard: it substitutes
# FMAXNM for the kernels' maxps selects [M], and FMAXNM returns the number where MAXPS returns its second operand,
# so every chunk holding a NaN would take a different maximum.
#
# The transcribed expf evaluates its polynomial as separate multiplies and adds, as the source it comes from does.
# Contracting those three sites is bit-neutral, measured over all 2^32 arguments; the flag holds the written form
# so that a build setting cannot silently choose a variant of the function, and the one site where the variants do
# part is written as an explicit fma instead, immune to the flag (runtime/src/basalt_cpu_expf.cpp, WHICH VARIANT,
# AND WHAT FUSES IN IT).
set_source_files_properties(src/basalt_cpu.cpp src/basalt_cpu_expf.cpp PROPERTIES COMPILE_FLAGS "-ffp-contract=off")
target_include_directories(_native PRIVATE src)
target_link_libraries(_native PRIVATE "-framework Metal" "-framework Foundation" "-framework IOKit")
set(LITHOS_METAL_NATIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/monolith/runtime" CACHE PATH "Python extension output directory")
set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${LITHOS_METAL_NATIVE_OUTPUT_DIRECTORY}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LITHOS_METAL_NATIVE_OUTPUT_DIRECTORY}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${LITHOS_METAL_NATIVE_OUTPUT_DIRECTORY}")
