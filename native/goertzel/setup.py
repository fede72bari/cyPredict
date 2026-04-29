# compile from console: python setup.py build_ext --inplace

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'goertzel',
        sources=['goertzel.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
    ),
]

setup(
    name='goertzel',
    version='1.2',
    description='Goertzel transform with pybind11 and NumPy 2.x compatibility',
    ext_modules=ext_modules,
)
