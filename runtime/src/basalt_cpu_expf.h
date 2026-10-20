// SPDX-License-Identifier: MIT
// Copyright (c) 2017-2025, Arm Limited (MIT OR Apache-2.0 WITH LLVM-exception; taken under MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from ARM-software/optimized-routines math/expf.c and math/exp2f_data.c @
// f67e7daa5457646fad9f75580263dba654c988aa (third_party/NOTICE, third_party/ARM-OPTIMIZED-ROUTINES-LICENSE), the
// source glibc's generic expf has carried since 2017 (sysdeps/ieee754/flt-32/e_expf.c); and from basalt
// (jesdga95/basalt) src/kernels/cpu/pool.cpp:434 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
// third_party/NOTICE): the SwiGLU expression the CPU expert pool applies, in its order.
//
// The one piece of arithmetic basalt's CPU expert pool performs itself. basalt runs on Linux with glibc, where the
// `std::exp` of a float at pool.cpp:434 is glibc's `expf`; Apple's `expf` is a different implementation and neither
// is correctly rounded, so calling the platform's cannot be bit-exact. These are the algorithm and the table
// instead, in portable C++.
#pragma once

namespace monolith::basalt_cpu {

/// glibc's `expf`, transcribed from the Arm source it is compiled from: 0.502 ulp worst case under round to
/// nearest, not correctly rounded. Every input is valid, specials included.
float expf_glibc(float x);

/// basalt's SwiGLU, `(g / (1.f + std::exp(-g))) * u` (pool.cpp:434) - the add, the divide and the multiply in the
/// order gcc emits them, each with the NaN x86 writes, which are both part of the result.
float swiglu(float g, float u);

}  // namespace monolith::basalt_cpu
