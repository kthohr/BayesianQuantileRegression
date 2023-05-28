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

/*
 * Gibbs sampler
 */

#ifndef _bqreg_sampler_HPP
#define _bqreg_sampler_HPP

#ifdef BQREG_USE_OPENMP
    #pragma omp declare reduction (+: ColVec_t: omp_out=omp_out+omp_in) \
        initializer(omp_priv = ColVec_t::Zero(omp_orig.size()))

    #pragma omp declare reduction (+: Mat_t: omp_out=omp_out+omp_in) \
        initializer(omp_priv = Mat_t::Zero(omp_orig.cols(),omp_orig.cols()))
#endif

inline
size_t
generate_seed_value(const int ind_inp, const int n_threads, rand_engine_t& rand_engine)
{
    return static_cast<size_t>( (stats::runif(fp_t(0), fp_t(1), rand_engine) + ind_inp + n_threads) * 1000 );
}

inline
void
qr_gibbs_iteration(
    const ColVec_t& Y,
    const Mat_t& X,
    const ColVec_t& prior_beta_mu, //  prior_beta_var_inv * prior_beta_mean
    const Mat_t& prior_beta_var_inv,
    const fp_t prior_sigma_shape,
    const fp_t prior_sigma_scale,
    const fp_t theta_par,
    const fp_t omega_sq_par,
    const bool keep_sigma_fixed, // keep sigma^2 value fixed for sampling
    const int omp_n_threads,
    ColVec_t& beta_draw,
    ColVec_t& nu_draw,
    fp_t& sigma_draw,
    std::vector<rand_engine_t>& rand_engines_vec
)
{
    (void)(omp_n_threads); // for !BQREG_USE_OPENMP case

    const size_t n = Y.size();
    const size_t K = X.cols();

    // draw beta

    Mat_t var_mat_sum = Mat_t::Zero(K,K);

#ifdef BQREG_USE_OPENMP
    #pragma omp parallel for num_threads(omp_n_threads) reduction(+:var_mat_sum)
#endif
    for (size_t i = 0; i < n; ++i) {
        var_mat_sum += X.row(i).transpose() * X.row(i) / ( omega_sq_par * sigma_draw * nu_draw(i) );
    }

    const Mat_t post_beta_var = ( var_mat_sum + prior_beta_var_inv ).inverse();

    //

    ColVec_t sum_vec = ColVec_t::Zero(K);

#ifdef BQREG_USE_OPENMP
    #pragma omp parallel for num_threads(omp_n_threads) reduction(+:sum_vec)
#endif
    for (size_t i = 0; i < n; ++i) {
        sum_vec += X.row(i) * ( Y(i) - theta_par * nu_draw(i) ) / ( omega_sq_par * sigma_draw * nu_draw(i) );
    }

    const ColVec_t post_beta_mean = post_beta_var * (sum_vec + prior_beta_mu);

    beta_draw = post_beta_mean + post_beta_var.llt().matrixL() * stats::rnorm<ColVec_t>(K, 1, fp_t(0), fp_t(1), rand_engines_vec[0]);

    // draw nu

    const fp_t gamma_par = std::sqrt( (2 / sigma_draw) + (theta_par * theta_par) / (sigma_draw * omega_sq_par) );
    const fp_t tmp_scale_val = std::sqrt( sigma_draw * omega_sq_par );

#ifdef BQREG_USE_OPENMP
    #pragma omp parallel for num_threads(omp_n_threads)
#endif
    for (size_t i = 0; i < n; ++i) {
        size_t thread_num = 0;

#ifdef BQREG_USE_OPENMP
        thread_num = omp_get_thread_num();
#endif

        const fp_t err_val = Y(i) - X.row(i).dot(beta_draw);
        const fp_t delta_par = std::abs(err_val) / tmp_scale_val;
        nu_draw(i) = fp_t(1) / stats::rinvgauss(gamma_par, delta_par, rand_engines_vec[thread_num]);
    }

    // draw sigma

    if (!keep_sigma_fixed) {
        fp_t sum_err_val = 0;

    #ifdef BQREG_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads) reduction(+:sum_err_val)
    #endif
        for (size_t i = 0; i < n; ++i) {
            const fp_t err_val = Y(i) - X.row(i).dot(beta_draw) - theta_par * nu_draw(i);
            sum_err_val += (err_val * err_val) / (omega_sq_par * nu_draw(i));
        }

        const fp_t post_sigma_shape_par = prior_sigma_shape + (3 * n / fp_t(2));
        const fp_t post_sigma_scale_par = (2 * prior_sigma_scale + 2 * nu_draw.array().sum() + sum_err_val ) / 2;

        sigma_draw = fp_t(1) / stats::rgamma(post_sigma_shape_par, 1 / post_sigma_scale_par, rand_engines_vec[0]);
    }
}

inline
void
qr_gibbs(
    const ColVec_t& Y,
    const Mat_t& X,
    const fp_t tau,
    const ColVec_t& beta_initial_draw,
    const ColVec_t& prior_beta_mean,
    const Mat_t& prior_beta_var,
    const fp_t prior_sigma_shape,
    const fp_t prior_sigma_scale,
    const size_t n_burnin_draws,
    const size_t n_keep_draws,
    const size_t thinning_factor,
    const bool keep_sigma_fixed,
    int omp_n_threads,
    Mat_t& beta_draws_storage,
    Mat_t& z_draws_storage,
    ColVec_t& sigma_draws_storage,
    rand_engine_t& rand_engine
)
{
#ifdef BQREG_USE_OPENMP
    if (omp_n_threads < 0) {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }

    if (omp_n_threads == 0) {
        omp_n_threads = 1;
    }
#else
    omp_n_threads = 1;
#endif

    //

    std::vector<rand_engine_t> rand_engines_vec;

    for (int i = 0; i < omp_n_threads; ++i) {
        size_t seed_val = generate_seed_value(i, omp_n_threads, rand_engine);
        rand_engines_vec.push_back(rand_engine_t(seed_val));
    }

    //

    const size_t n_total_draws = n_burnin_draws + (thinning_factor + 1) * n_keep_draws;

    const size_t n = Y.size();
    const size_t K = X.cols();

    const fp_t theta_par = (1 - 2 * tau) / (tau * (1 - tau));
    const fp_t omega_sq_par = 2 / (tau * (1 - tau));

    const Mat_t prior_beta_var_inv = prior_beta_var.inverse();
    const Mat_t prior_beta_mu = prior_beta_var_inv * prior_beta_mean;

    // set storage containers

    beta_draws_storage.setZero(K, n_keep_draws);
    z_draws_storage.setZero(n, n_keep_draws);
    sigma_draws_storage.setZero(n_keep_draws);

    // set initial values for the draws

    ColVec_t beta_draw = beta_initial_draw;

    fp_t sigma_draw = (Y - X * beta_draw).array().pow(2).sum() / fp_t(n);

    ColVec_t nu_draw = ColVec_t::Constant(n, sigma_draw);

    if (keep_sigma_fixed) {
        sigma_draw = fp_t(1);
    }

    // main loop

    size_t mcmc_save_ind = 0;

    for (size_t mcmc_ind = 0; mcmc_ind < n_total_draws; ++mcmc_ind) {
        
        // one iteration of gibbs sampling

        qr_gibbs_iteration(Y, 
                           X, 
                           prior_beta_mu, 
                           prior_beta_var_inv,
                           prior_sigma_shape,
                           prior_sigma_scale,
                           theta_par,
                           omega_sq_par,
                           keep_sigma_fixed,
                           omp_n_threads,
                           beta_draw,
                           nu_draw,
                           sigma_draw,
                           rand_engines_vec);
        
        // save draws

        if (mcmc_ind >= n_burnin_draws && (mcmc_ind - n_burnin_draws) % (thinning_factor + 1) == 0 ) {
            // note: (mcmc_ind - n_burnin_draws) could underflow but...
            // the second condition will not be checked if the first condition does not pass

            beta_draws_storage.col(mcmc_save_ind) = beta_draw;
            z_draws_storage.col(mcmc_save_ind) = nu_draw / sigma_draw;
            sigma_draws_storage(mcmc_save_ind) = sigma_draw;

            ++mcmc_save_ind;
        }
    }
}

#endif
