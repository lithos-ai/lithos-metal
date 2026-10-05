"""Build a self-contained wheel, including the Metal runtime on macOS."""
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent.resolve()


class BuildPython(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "monolith" / "_kernels"
        shutil.copytree(ROOT / "kernels", destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("README.md", "__pycache__"))


class BuildMetal(build_ext):
    def build_extension(self, ext):
        import cmake
        cmake_bin = str(Path(cmake.CMAKE_BIN_DIR) / 'cmake')
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / "metal"
        subprocess.run([cmake_bin, "-S", str(ROOT), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}",
                        f"-DLITHOS_METAL_NATIVE_OUTPUT_DIRECTORY={output}"], check=True)
        subprocess.run([cmake_bin, "--build", str(build), "--config", "Release", "--parallel", "4"], check=True)


setup(cmdclass={"build_py": BuildPython, "build_ext": BuildMetal},
      ext_modules=[Extension("monolith.runtime._native", sources=[])] if platform.system() == "Darwin" else [])
