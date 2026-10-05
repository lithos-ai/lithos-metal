#!/bin/bash
# Run the probe suite on another Apple-silicon Mac over SSH and fetch the results file.
#   ./remote_run.sh user@m4-host                 everything
#   ./remote_run.sh ec2-user@1.2.3.4 p10_claim_protocol p11_interop_overlap
# The remote machine needs only SSH access and the Xcode Command Line Tools. It must be BARE METAL: virtualized macOS
# (GitHub/CircleCI/Tart/Anka runners) exposes a paravirtual GPU whose scheduling says nothing about the real one.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <ssh-host> [probe ...]"; exit 1; }
host=$1; shift
here="$(cd "$(dirname "$0")" && pwd)"
ssh "$host" 'sysctl -n machdep.cpu.brand_string; sw_vers -productVersion; command -v clang >/dev/null || echo "WARNING: clang missing (see run_all.sh header)"'
ssh "$host" 'mkdir -p ~/gpu-probes'
rsync -az --delete --exclude build --exclude results "$here/" "$host:gpu-probes/"
ssh "$host" "cd ~/gpu-probes && ./run_all.sh $*"
mkdir -p "$here/results"
rsync -az "$host:gpu-probes/results/" "$here/results/"
echo "fetched into $here/results/"; ls -t "$here/results" | head -3
