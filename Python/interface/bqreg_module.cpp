/*################################################################################
  ##
  ##   Copyright (C) 2021-2022 Keith O'Hara
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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "bqreg.hpp"
#include "bqreg_module_class.hpp"

PYBIND11_MODULE(bqreg_wrapper, m)
{
    pybind11::class_<bqreg_module_Py>(m, "bqreg")
        .def(pybind11::init<>())

        //

        .def( "get_omp_n_threads", &bqreg_module_Py::get_omp_n_threads )
        .def( "set_omp_n_threads", &bqreg_module_Py::set_omp_n_threads )

        .def( "set_seed_value", &bqreg_module_Py::set_seed_value )

        .def( "load_data", &bqreg_module_Py::load_data )
        .def( "set_quantile_target", &bqreg_module_Py::set_quantile_target )
        .def( "set_prior_params", &bqreg_module_Py::set_prior_params )

        .def( "set_initial_beta_draw", &bqreg_module_Py::set_initial_beta_draw )

        .def( "gibbs", &bqreg_module_Py::gibbs )
    ;
}
