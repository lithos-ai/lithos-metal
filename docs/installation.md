# Installing and releasing lithos-metal

lithos-metal runs on Apple silicon with macOS 26+. Use a registered chip backend and sufficient
memory for the target, draft, and context. The two large NVFP4 target/draft combinations
have been exercised on a 40-core M5 Max with 48 GB unified memory.

## Install from source

Install Xcode Command Line Tools (`xcode-select --install`) and Python 3.12, then:

```bash
git clone https://github.com/lithos-ai/lithos-metal.git
cd lithos-metal
python3.12 -m venv .venv
source .venv/bin/activate
pip install '.[serve]'
lithos-metal --version
lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4
```

The package build compiles the native extension and bundles all common/chip Metal kernel
sources and recipes. An installed wheel works outside the source checkout. An editable
install (`pip install -e '.[serve]'`) is available for development. The internal `monolith`
Python namespace and `python -m monolith.serve` remain supported.

## Homebrew release

Install the published package from the [Lithos tap](https://github.com/lithos-ai/homebrew-tap):

```bash
brew install lithos-ai/tap/lithos-metal
lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4
```

The tap's `Formula/lithos-metal.rb` pins the archive and SHA-256 from the
[GitHub release](https://github.com/lithos-ai/lithos-metal/releases/latest). Tap CI runs
`brew install` and `brew test` on macOS ARM64.

On an Apple silicon Mac using Python 3.12:

```bash
python tools/release/build_macos.py
```

This creates `dist/lithos-metal-VERSION-macos-arm64.tar.gz` with the compiled lithos-metal wheel and every
serving dependency as a wheel, plus `dist/lithos-metal.rb`. The formula's archive URL and SHA-256
are generated from that exact bundle. Homebrew supplies Python 3.12; installation inside
its private virtual environment uses `--no-index` and requires no compiler or model
weights. Metal shader specialization still occurs at inference time.

The `macOS release` GitHub workflow builds and smoke-tests the bundle on `macos-26`.
Manual runs upload reviewable artifacts. Pushing a `vVERSION` tag that matches
`pyproject.toml` publishes the bundle and formula as GitHub release assets. Use
`--repository OWNER/REPO` for a relocated release repository.

To publish the tap, copy the generated formula into `Formula/lithos-metal.rb` in
`lithos-ai/homebrew-tap` after publishing the matching release. Test with
`brew install lithos-ai/tap/lithos-metal` and `brew test lithos-ai/tap/lithos-metal`. Do not publish a formula
that points at an unbuilt version or invent a checksum. Bump the package version for
subsequent releases and regenerate the bundle and formula together.
