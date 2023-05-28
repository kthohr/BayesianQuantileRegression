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

#ifndef _R_bqreg_module_class_HPP
#define _R_bqreg_module_class_HPP

using namespace bqreg;

class bqreg_module_R
{
    public:
        ColVec_t Y;
        Mat_t X;

        fp_t tau;

        ColVec_t prior_beta_mean;
        Mat_t prior_beta_var;
        fp_t prior_sigma_shape;
        fp_t prior_sigma_scale;

        //

        SEXP get_omp_n_threads();
        void set_omp_n_threads(const int omp_n_threads_inp);

        void set_seed_value(const size_t seed_val_inp);

        void load_data(const ColVec_t& Y_inp, const Mat_t& X_inp);
        void set_quantile_target(const fp_t tau_inp);
        void set_prior_params(const ColVec_t& prior_beta_mean_inp, const Mat_t& prior_beta_var_inp, const fp_t prior_sigma_shape_inp, const fp_t prior_sigma_scale_inp);

        SEXP get_initial_beta_draw();
        void set_initial_beta_draw(const ColVec_t& beta_initial_draw_inp);

        SEXP gibbs(const size_t n_burnin_draws, const size_t n_keep_draws, const size_t thinning_factor);
    
    private:
        bool keep_sigma_fixed = false;
        int omp_n_threads = -1;
        rand_engine_t rand_engine = rand_engine_t(std::random_device{}());

        ColVec_t beta_initial_draw;
};

#include "bqreg_R_module_fns.hpp"

#endif
