
from setuptools import setup, Extension
import numpy
import pybind11

ext_modules = [
    Extension(
        name='cyfitness',
        sources=['cyfitness.cpp'],
        include_dirs=[numpy.get_include(), pybind11.get_include()],
        language='c++'
    )
]

setup(
    name='cyfitness',
    version='1.0',
    description='CDC Fitness function (GA) in C++',
    ext_modules=ext_modules
)
