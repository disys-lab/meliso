from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("build.meliso",
                  sources=[ "src/cython/mel.pyx"],
                  libraries=["gomp","mlp"],
                  language="c++",                   # remove this if C and not C++
                  extra_compile_args=["-c", "-g","-fPIC","-fopenmp", "-O3",'-std=c++0x','-w', '-Isrc/mlp_neurosim/', '-Isrc/cython/'], #"-L./build/mlp_neurosim","-lmlp"],
                  extra_link_args=["-lgomp","-L./build/","-lmlp"]
             )
        ]
)