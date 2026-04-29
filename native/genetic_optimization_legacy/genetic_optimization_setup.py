# run: python genetic_optimization_setup.py build_ext --inplace

from setuptools import setup, Extension
import numpy
import pybind11

include_dirs = [
    numpy.get_include(),
    pybind11.get_include()
]

ext_modules = [
    Extension(
        name='genetic_optimization',
        sources=['genetic_optimization.c'],
        include_dirs=include_dirs,
        language='c',  # o 'c++' se usi file .cpp
        extra_compile_args=[],
    )
]

setup(
    name='GeneticOptimization',
    version='2.0',
    description='C extension for genetic optimization, compatibile with numpy 2.X',
    ext_modules=ext_modules,
)


