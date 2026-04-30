from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def compile_args():
    import sys

    if sys.platform == "win32":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]


setup(
    name="cyGAopt",
    version="2.0.0",
    ext_modules=[
        Pybind11Extension(
            "cyGAopt",
            [str(Path("cyGAopt.cpp"))],
            include_dirs=[str(Path("."))],
            extra_compile_args=compile_args(),
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
