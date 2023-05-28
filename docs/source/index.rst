.. Copyright (c) 2021-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Introduction
============

The Bayesian Quantile Regression (BQReg) library is a lightweight C++ implementation of Bayesian quantile regression using the asymmetric Laplace density representation of the problem, with bindings for Python and R.

Features:

- A header-only C++11 library with OpenMP-accelerated MCMC sampling for parallel computation.

- Built on the `Eigen <http://eigen.tuxfamily.org/index.php>`_ (version >= 3.4.0) templated linear algebra library for fast and efficient matrix-based computation.

- Straightforward linking with parallelized BLAS libraries, such as `OpenBLAS <https://github.com/xianyi/OpenBLAS>`_.

- Seamless integration with Python and R through Pybind11 and Rcpp modules.

- Available as a single precision (``float``) or double precision (``double``) library.

- Released under a permissive, non-GPL license.

Author: Keith O'Hara

License: Apache Version 2.0

----

Installation
------------

The C++ implementation of BQReg is a header-only library. First, clone the library and related submodules:

.. code:: bash

    # clone optim into the current directory
    git clone https://github.com/kthohr/BayesianQuantileRegression ./BayesianQuantileRegression

    # change directory
    cd ./BayesianQuantileRegression

    # clone necessary submodules
    git submodule update --init


Then simply add the header files (found under `cpp/include`) to your project using:

.. code:: cpp

    #include "bqreg.hpp"


The only other dependencies are a copy of `Eigen <http://eigen.tuxfamily.org/index.php>`_ (version >= 3.4.0) and a C++14-compatible compiler.

For detailed instructions on how to install the R and Python bindings, see the :ref:`installation page <installation>`.

----

Contents
--------

.. toctree::
   :caption: Guide
   :maxdepth: 2
   
   installation
   model_description

.. toctree::
   :caption: API
   :maxdepth: 2
   
   cpp
   python
   r

