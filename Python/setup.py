import os
import sys

from setuptools import Extension, setup

EIGEN_INCLUDE_PATH = os.environ.get("EIGEN_INCLUDE_PATH",1)
PYBIND11_INCLUDE_PATH = os.environ.get("PYBIND11_INCLUDE_PATH",1)

#

cpp_compile_args = [
    "-std=c++14",
    "-stdlib=libc++",
    "-ffp-contract=fast",
    "-fopenmp"
]

cpp_linking_args = [
    "-fopenmp"
]

#

ext_modules = [
    Extension(
        "bqreg_wrapper",
        ["interface/bqreg_py_module.cpp"],
        include_dirs=[EIGEN_INCLUDE_PATH, PYBIND11_INCLUDE_PATH, "../extr/gcem/include", "../extr/stats/include", "../cpp/include"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=cpp_linking_args
    )
]

#

setup(
    name="pybqreg",
    version="0.3.0",
    author="Keith O'Hara",
    author_email="kth.ohr@gmail.com",
    description="Bayesian Quantile Regression",
    url='https://github.com/kthohr/BayesianQuantileRegression',
    ext_modules=ext_modules,
    packages=["pybqreg"]
)
