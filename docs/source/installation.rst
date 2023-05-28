.. Copyright (c) 2021-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

.. _installation:

Installation
============

The C++ implementation of BQReg is a header-only library, and we provide bindings for Python and R. 


C++
---

The C++ library requires only the Eigen(3) C++ linear algebra library, as well as the submodules included in the source code. (Note that Eigen version 3.4.0 requires a C++14-compatible compiler.)

First clone the library and required submodules:

.. code:: bash

    # clone optim into the current directory
    git clone https://github.com/kthohr/BayesianQuantileRegression ./BayesianQuantileRegression

    # change directory
    cd ./BayesianQuantileRegression

    # clone necessary submodules
    git submodule update --init

Then simply add the header files (found under ``cpp/include``) to your project using:

.. code:: cpp

    #include "bqreg.hpp"

The following options should be declared **before** including the BQReg header files. 

- OpenMP functionality is enabled by default if the ``_OPENMP`` macro is detected (e.g., by invoking ``-fopenmp`` with GCC or Clang). 

  - To explicitly enable OpenMP features, use:

  .. code:: cpp

    #define BQREG_USE_OPENMP

  - To explicitly disable OpenMP functionality, use:

  .. code:: cpp

    #define BQREG_DONT_USE_OPENMP


Python
------

If you do not already have Eigen3 and Pybind11, clone and set environment variables:

.. code:: bash

    # clone Eigen3
    git clone https://gitlab.com/libeigen/eigen.git

    # clone Pybind11
    git clone https://github.com/pybind/pybind11.git

    # set environment variables
    export EIGEN_INCLUDE_PATH = "<path to this directory>/eigen"
    export PYBIND11_INCLUDE_PATH = "<path to this directory>/pybind11/include"

Then install from the Python subdirectory:

.. code:: bash

    # change directory into the Python subdirectory
    cd BayesianQuantileRegression/Python

    # install the package
    python3 -m pip install . --user

R
-

First install the ``RcppEigen`` package from R:

.. code:: R

    install.packages("RcppEigen")

Then install the package from the ``R`` subdirectory:

.. code:: bash

    # change directory into the R subdirectory
    cd BayesianQuantileRegression/R

    # install the package
    R CMD INSTALL .


----
