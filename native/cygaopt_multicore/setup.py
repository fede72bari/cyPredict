from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def compile_args():
    import sys

    if sys.platform == "win32":
        return ["/O2", "/std:c++17", "/openmp"]
    return ["-O3", "-std=c++17", "-fopenmp"]


def link_args():
    import sys

    if sys.platform == "win32":
        return []
    return ["-fopenmp"]


setup(
    name="cyGAoptMultiCore",
    version="2.0.0",
    ext_modules=[
        Pybind11Extension(
            "cyGAoptMultiCore",
            [str(Path("cyGAoptMulticore.cpp"))],
            include_dirs=[str(Path("..") / "cygaopt")],
            extra_compile_args=compile_args(),
            extra_link_args=link_args(),
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
