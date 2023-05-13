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

#include "bqreg.hpp"
#include "bqreg_R_module_class.hpp"

RCPP_MODULE(bqreg_wrapper)
{
    using namespace Rcpp;
    
    class_<bqreg_module_R>("bqreg")
        .default_constructor()

        //

        .method( "get_omp_n_threads", &bqreg_module_R::get_omp_n_threads )
        .method( "set_omp_n_threads", &bqreg_module_R::set_omp_n_threads )

        .method( "set_seed_value", &bqreg_module_R::set_seed_value )

        .method( "load_data", &bqreg_module_R::load_data )
        .method( "set_quantile_target", &bqreg_module_R::set_quantile_target )
        .method( "set_prior_params", &bqreg_module_R::set_prior_params )

        .method( "set_initial_beta_draw", &bqreg_module_R::set_initial_beta_draw )

        .method( "gibbs", &bqreg_module_R::gibbs )
    ;
}
