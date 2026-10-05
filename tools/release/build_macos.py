#!/usr/bin/env python3
"""Build an offline Homebrew wheel bundle and its checksum-pinned formula on Apple silicon."""
import argparse
import hashlib
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'dist')
    parser.add_argument('--repository', default='lithos-ai/lithos-metal', help='GitHub release repository')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
        parser.error('Build with Python 3.12 on an Apple silicon Mac')
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lithos-metal-release-') as tmp:
        bundle = Path(tmp) / f'lithos-metal-{version}'
        wheels = bundle / 'wheels'
        wheels.mkdir(parents=True)
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps', '--wheel-dir', str(wheels)], check=True)
        wheel = next(wheels.glob('lithos_metal-*.whl'))
        subprocess.run([sys.executable, '-m', 'pip', 'download', f'{wheel}[serve]', '--only-binary=:all:',
                        '--dest', str(wheels)], check=True)
        shutil.copy2(ROOT / 'LICENSE', bundle / 'LICENSE')
        shutil.copy2(ROOT / 'README.md', bundle / 'README.md')
        (bundle / 'SHA256SUMS').write_text(''.join(
            f'{hashlib.sha256(p.read_bytes()).hexdigest()}  wheels/{p.name}\n' for p in sorted(wheels.iterdir())))
        name = f'lithos-metal-{version}-macos-arm64.tar.gz'
        archive = output / name
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(bundle, arcname=bundle.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'lithos-metal.rb').write_text(formula(version, args.repository, name, digest))
    print(f'{archive}\nSHA256 {digest}\nFormula: {output / "lithos-metal.rb"}')


def formula(version, repository, archive, digest):
    return f'''class LithosMetal < Formula
  desc "lithos-metal: local LLM inference on Apple silicon"
  homepage "https://github.com/{repository}"
  url "https://github.com/{repository}/releases/download/v{version}/{archive}"
  version "{version}"
  sha256 "{digest}"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :tahoe
  depends_on "python@3.12"

  def install
    python = Formula["python@3.12"].opt_bin/"python3.12"
    system python, "-m", "venv", libexec
    wheel = Dir[buildpath/"wheels/lithos_metal-*.whl"].first
    system libexec/"bin/python", "-m", "pip", "install", "--no-index",
           "--find-links=#{{buildpath}}/wheels", "#{{wheel}}[serve]"
    bin.install_symlink libexec/"bin/lithos-metal"
  end

  test do
    assert_match "lithos-metal", shell_output("#{{bin}}/lithos-metal --version")
    assert_match "DSpark", shell_output("#{{bin}}/lithos-metal models")
    system libexec/"bin/python", "-c",
           "from monolith.runtime import is_available; assert is_available()"
  end
end
'''


if __name__ == '__main__':
    main()
