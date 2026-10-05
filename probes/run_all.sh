#!/bin/bash
# Build and run the Apple-GPU probes on THIS machine; results are tee'd to results/<chip>_<cores>c_macOS<ver>_<time>.txt.
#   ./run_all.sh                       everything (~3-4 min; ~10 GB of free RAM for the 3 GB streaming buffers)
#   ./run_all.sh p10_claim_protocol p11_interop_overlap     a subset
#   GPU_CORES=20 ./run_all.sh          override the auto-detected GPU core count
# Needs only the Xcode Command Line Tools (shaders are compiled at runtime). If `clang` is missing on a headless box:
#   touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
#   softwareupdate -i "$(softwareupdate -l | grep -o 'Command Line Tools for Xcode-[0-9.]*' | tail -1)"
# NOTE: Apple GPUs do not preempt a running dispatch (p6). Every probe is bounded; the longest single dispatch is ~1.5 s
# (p6), so expect brief display stalls on a machine with a screen. Never add an unbounded spin to a probe.
set -uo pipefail
cd "$(dirname "$0")"
command -v clang >/dev/null || { echo "clang not found - install the Xcode Command Line Tools (see header)"; exit 1; }
mkdir -p build results
ALL="p1_limits p2_sync p3_residency p4_core_model p5_bandwidth p5b_access_pattern p6_preemption p6b_interleave p7_dispatch_overhead p8_threadgroup_mem p9_clock_warp p10_claim_protocol p11_interop_overlap p12_stream_geometry p13_decode_gemv p14_tensor_ops p15_frame_pacing"
chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr ' ' '-')
cores=${GPU_CORES:-$(ioreg -l 2>/dev/null | awk -F'= ' '/"gpu-core-count"/{print $2; exit}')}
out="results/${chip:-unknown}_${cores:-NA}c_macOS$(sw_vers -productVersion)_$(date +%Y%m%d-%H%M%S).txt"
main() {
  echo "# chip: $(sysctl -n machdep.cpu.brand_string)   gpu-cores: ${cores:-?}   ram: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  echo "# os: macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))   date: $(date -u +%Y-%m-%dT%H:%M:%SZ)   on-battery: $(pmset -g batt 2>/dev/null | grep -c 'Battery Power')"
  for src in $ALL; do
    if [[ $# -gt 0 && " $* " != *" $src "* ]]; then continue; fi
    if ! clang -fobjc-arc -O2 -Wno-everything -framework Foundation -framework Metal -framework CoreGraphics -framework IOKit -framework Cocoa -framework QuartzCore "$src.m" -o "build/$src"; then echo "BUILD FAILED: $src"; continue; fi
    echo; echo "################ $src ################"
    "./build/$src" || echo "PROBE FAILED: $src (exit $?)"
  done
}
main "$@" 2>&1 | tee "$out"
echo; echo "results saved to probes/$out"
