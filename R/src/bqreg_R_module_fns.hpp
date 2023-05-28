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

#ifndef _R_bqreg_module_fns_HPP
#define _R_bqreg_module_fns_HPP

SEXP
inline
bqreg_module_R::get_omp_n_threads()
{
    try {
        return Rcpp::wrap(this->omp_n_threads);
    } catch( std::exception &ex ) {
        forward_exception_to_r( ex );
    } catch(...) {
        ::Rf_error( "bqreg: C++ exception (unknown reason)" );
    }
    return R_NilValue;
}

void
inline
bqreg_module_R::set_omp_n_threads(const int omp_n_threads_inp)
{
    this->omp_n_threads = omp_n_threads_inp;
}

void
inline
bqreg_module_R::set_seed_value(const size_t seed_val_inp)
{
    this->rand_engine = rand_engine_t(seed_val_inp);
}

void
inline
bqreg_module_R::load_data(const ColVec_t& Y_inp, const Mat_t& X_inp)
{
    this->Y = Y_inp;
    this->X = X_inp;

    this->beta_initial_draw.setZero(X.cols());
}

void
inline
bqreg_module_R::set_quantile_target(const fp_t tau_inp)
{
    this->tau = tau_inp;
}

void
inline
bqreg_module_R::set_prior_params(
    const ColVec_t& prior_beta_mean_inp, 
    const Mat_t& prior_beta_var_inp, 
    const fp_t prior_sigma_shape_inp, 
    const fp_t prior_sigma_scale_inp
)
{
    this->prior_beta_mean = prior_beta_mean_inp;
    this->prior_beta_var = prior_beta_var_inp;
    this->prior_sigma_shape = prior_sigma_shape_inp;
    this->prior_sigma_scale = prior_sigma_scale_inp;
}

SEXP
inline
bqreg_module_R::get_initial_beta_draw()
{
    try {
        return Rcpp::wrap(this->beta_initial_draw);
    } catch( std::exception &ex ) {
        forward_exception_to_r( ex );
    } catch(...) {
        ::Rf_error( "bqreg: C++ exception (unknown reason)" );
    }
    return R_NilValue;
}

void
inline
bqreg_module_R::set_initial_beta_draw(
    const ColVec_t& beta_initial_draw_inp
)
{
    this->beta_initial_draw = beta_initial_draw_inp;
}

SEXP
inline
bqreg_module_R::gibbs(
    const size_t n_burnin_draws, 
    const size_t n_keep_draws,
    const size_t thinning_factor
)
{
    try {
        Mat_t beta_draws;
        Mat_t z_draws;
        ColVec_t sigma_draws;

        if (beta_initial_draw.size() != X.cols()) {
            beta_initial_draw.setZero(X.cols());
        }

        qr_gibbs(Y,
                 X,
                 tau,
                 beta_initial_draw,
                 prior_beta_mean,
                 prior_beta_var,
                 prior_sigma_shape,
                 prior_sigma_scale,
                 n_burnin_draws,
                 n_keep_draws,
                 thinning_factor,
                 keep_sigma_fixed,
                 omp_n_threads,
                 beta_draws,
                 z_draws,
                 sigma_draws,
                 rand_engine);

        return Rcpp::List::create(Rcpp::Named("beta_draws") = beta_draws, 
                                  Rcpp::Named("z_draws") = z_draws, 
                                  Rcpp::Named("sigma_draws") = sigma_draws);
    } catch( std::exception &ex ) {
        forward_exception_to_r( ex );
    } catch(...) {
        ::Rf_error( "bqreg: C++ exception (unknown reason)" );
    }
    return R_NilValue;
}

#endif
