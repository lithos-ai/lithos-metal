"""Locate kernel sources in either a source checkout or an installed wheel."""
from pathlib import Path


def kernel_root():
    package = Path(__file__).resolve().parent
    bundled = package / "_kernels"
    return bundled if bundled.is_dir() else package.parent / "kernels"
