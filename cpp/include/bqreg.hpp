/*################################################################################
  ##
  ##   Copyright (C) 2021-2023 Keith O'Hara
  ##
  ##   This file is part of the BayesianQuantileRegression library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

#ifndef _bqreg_HPP
#define _bqreg_HPP

#include <numeric>
#include <random>

// version

#ifndef BQREG_VERSION_MAJOR
    #define BQREG_VERSION_MAJOR 0
#endif

#ifndef BQREG_VERSION_MINOR
    #define BQREG_VERSION_MINOR 2
#endif

#ifndef BQREG_VERSION_PATCH
    #define BQREG_VERSION_PATCH 0
#endif

//

#ifdef _MSC_VER
    #error bqreg: MSVC is not supported
#endif

//

#if defined(_OPENMP) && !defined(BQREG_DONT_USE_OPENMP)
    #undef BQREG_USE_OPENMP
    #define BQREG_USE_OPENMP
#endif

#if !defined(_OPENMP) && defined(BQREG_USE_OPENMP)
    #undef BQREG_USE_OPENMP

    #undef BQREG_DONT_USE_OPENMP
    #define BQREG_DONT_USE_OPENMP
#endif

// #ifdef BQREG_USE_OPENMP
    // #include "omp.h" //  OpenMP
// #endif

#ifdef BQREG_DONT_USE_OPENMP
    #ifdef BQREG_USE_OPENMP
        #undef BQREG_USE_OPENMP
    #endif
#endif

// floating point number type

#ifndef BQREG_FPN_TYPE
    #define BQREG_FPN_TYPE double
#endif

//

#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    #define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#endif

#ifdef BQREG_USE_RCPP_EIGEN
    #include <RcppEigen.h>
#else
    #include <Eigen/Dense>
#endif

#define STATS_ENABLE_EIGEN_WRAPPERS

#include "stats.hpp"

namespace bqreg
{
    using rand_engine_t = std::mt19937_64;
    using fp_t = BQREG_FPN_TYPE;

    using Mat_t = Eigen::Matrix<fp_t, Eigen::Dynamic, Eigen::Dynamic>;
    using ColVec_t = Eigen::Matrix<fp_t, Eigen::Dynamic, 1>;

    #include "bqreg_sampler.hpp"
}

#endif