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

#ifndef _bqreg_class_HPP
#define _bqreg_class_HPP

/**
 * Bayesian Quantile Regression class
 */

class bqreg_t
{
    public:
        ColVec_t Y;               /*!< An n x 1 vector defining the target variable */
        Mat_t X;                  /*!< An n x K matrix of features */

        fp_t tau;                 /*!< The target quantile value */

        ColVec_t prior_beta_mean; /*!< Mean of the prior distribution for \f$ \beta \f$ */
        Mat_t prior_beta_var;     /*!< Variance of the prior distribution for \f$ \beta \f$ */
        fp_t prior_sigma_shape;   /*!< Shape parameter of the prior distribution for \f$ \sigma \f$ */
        fp_t prior_sigma_scale;   /*!< Scale parameter of the prior distribution for \f$ \sigma \f$ */

        //

        /**
         * Default destructor.
         */

        ~bqreg_t() = default;

        /**
         * Constructor using a \c bqreg_t object to copy from
         *
         * @param obj_inp an object of type \c bqreg_t whose elements will be used to initalize a new object of the same type.
         */

        bqreg_t(const bqreg_t& obj_inp);

        /**
         * Constructor using a moveable \c bqreg_t object
         *
         * @param obj_inp a moveable object of type \c bqreg_t whose elements will be used to initalize a new object of the same type.
         */

        bqreg_t(bqreg_t&& obj_inp);

        /**
         * Constructor using data matrices
         *
         * @param Y_inp an n x 1 vector defining the target variable
         * @param X_inp an n x K matrix of features
         */

        explicit bqreg_t(const ColVec_t& Y_inp, const Mat_t& X_inp);

        /**
         * Constructor using moveable data matrices
         *
         * @param Y_inp an n x 1 vector defining the target variable
         * @param X_inp an n x K matrix of features
         */
        
        explicit bqreg_t(ColVec_t&& Y_inp, Mat_t&& X_inp);

        /**
         * Assignment operator
         *
         * @param obj_inp an object of type \c bqreg_t whose elements will be used to initalize a new object of the same type.
         * @return an object of type \c bqreg_t
         */

        bqreg_t& operator=(const bqreg_t& obj_inp);

        /**
         * Assignment operator
         *
         * @param obj_inp a moveable object of type \c bqreg_t whose elements will be used to initalize a new object of the same type.
         * @return an object of type \c bqreg_t
         */

        bqreg_t& operator=(bqreg_t&& obj_inp);

        //

        /**
         * OpenMP threads check
         *
         * @return the number of OpenMP threads to use in the Gibbs sampler stage.
         */

        int get_omp_n_threads();

        /**
         * OpenMP threads set
         *
         * @param omp_n_threads_inp the number of OpenMP threads to use in the Gibbs sampler stage.
         */
        
        void set_omp_n_threads(const int omp_n_threads_inp);

        /**
         * RNG engine seeding
         *
         * @param seed_val_inp the seed value for the RNG engine.
         */

        void set_seed_value(const size_t seed_val_inp);

        /**
         * Load data
         *
         * @param Y_inp an n x 1 vector defining the target variable
         * @param X_inp an n x K matrix of features
         */

        void load_data(const ColVec_t& Y_inp, const Mat_t& X_inp);

        /**
         * Set the target quantile value
         *
         * @param tau_inp a real value between zero and one
         */
        
        void set_quantile_target(const fp_t tau_inp);

        /**
         * Set the Prior
         * @brief Set the parameters of the distributions that define the prior
         *
         * @param prior_beta_mean_inp mean of the prior distribution for \f$ \beta \f$
         * @param prior_beta_var_inp variance of the prior distribution for \f$ \beta \f$
         * @param prior_sigma_shape_inp shape parameter of the prior distribution for \f$ \sigma \f$
         * @param prior_sigma_scale_inp scale parameter of the prior distribution for \f$ \sigma \f$
         */
        
        void set_prior_params(const ColVec_t& prior_beta_mean_inp, const Mat_t& prior_beta_var_inp, const fp_t prior_sigma_shape_inp, const fp_t prior_sigma_scale_inp);

        /**
         * Initial value check for the Gibbs sampler
         *
         * @return a vector defining the initial values for \f$ \beta \f$ in the Gibbs sampler.
         */

        ColVec_t get_initial_beta_draw();

        /**
         * Set the initial values for the Gibbs sampler
         *
         * @param beta_initial_draw_inp a vector defining the initial values for \f$ \beta \f$ in the Gibbs sampler.
         */

        void set_initial_beta_draw(const ColVec_t& beta_initial_draw_inp);

        /**
         * Run the Gibbs sampler
         *
         * @param n_burnin_draws the number of burnin draws
         * @param n_keep_draws the number of draws to keep, post burnin
         * @param thinning_factor the number of draws to skip between keep draws
         * @param beta_draws a writable matrix to store the draws of \f$ \beta \f$
         * @param z_draws a writable matrix to store the draws of \f$ z \f$
         * @param sigma_draws a writable vector to store the draws of \f$ \sigma \f$
         */

        void gibbs(const size_t n_burnin_draws, const size_t n_keep_draws, const size_t thinning_factor, Mat_t& beta_draws, Mat_t& z_draws, ColVec_t& sigma_draws);
    
    private:
        bool keep_sigma_fixed = false;
        int omp_n_threads = -1;
        rand_engine_t rand_engine = rand_engine_t(std::random_device{}());

        ColVec_t beta_initial_draw;
};

// member functions

inline
bqreg_t::bqreg_t(
    const ColVec_t& Y_inp, 
    const Mat_t& X_inp
)
{
    Y = Y_inp;
    X = X_inp;
}

inline
bqreg_t::bqreg_t(
    ColVec_t&& Y_inp, 
    Mat_t&& X_inp
)
{
    Y = std::move(Y_inp);
    X = std::move(X_inp);
}

//

inline
bqreg_t&
bqreg_t::operator=(const bqreg_t& obj_inp)
{
    Y = obj_inp.Y;
    X = obj_inp.X;

    prior_beta_mean   = obj_inp.prior_beta_mean;
    prior_beta_var    = obj_inp.prior_beta_var;
    prior_sigma_shape = obj_inp.prior_sigma_shape;
    prior_sigma_scale = obj_inp.prior_sigma_scale;

    // could also set/copy omp_n_threads, keep_sigma_fixed, rand_engine, and beta_initial_draw...

    return *this;
}

inline
bqreg_t&
bqreg_t::operator=(bqreg_t&& obj_inp)
{
    Y = std::move(obj_inp.Y);
    X = std::move(obj_inp.X);

    prior_beta_mean = std::move(obj_inp.prior_beta_mean);
    prior_beta_var  = std::move(obj_inp.prior_beta_var);
    prior_sigma_shape = obj_inp.prior_sigma_shape;
    prior_sigma_scale = obj_inp.prior_sigma_scale;

    // could also set/copy omp_n_threads, keep_sigma_fixed, rand_engine, and beta_initial_draw...

    return *this;
}

//

int
inline
bqreg_t::get_omp_n_threads()
{
    return this->omp_n_threads;
}

void
inline
bqreg_t::set_omp_n_threads(const int omp_n_threads_inp)
{
    this->omp_n_threads = omp_n_threads_inp;
}

void
inline
bqreg_t::set_seed_value(const size_t seed_val_inp)
{
    this->rand_engine = rand_engine_t(seed_val_inp);
}

void
inline
bqreg_t::load_data(const ColVec_t& Y_inp, const Mat_t& X_inp)
{
    this->Y = Y_inp;
    this->X = X_inp;
}

void
inline
bqreg_t::set_quantile_target(const fp_t tau_inp)
{
    this->tau = tau_inp;
}

void
inline
bqreg_t::set_prior_params(
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

ColVec_t
inline
bqreg_t::get_initial_beta_draw()
{
    return this->beta_initial_draw;
}

void
inline
bqreg_t::set_initial_beta_draw(
    const ColVec_t& beta_initial_draw_inp
)
{
    this->beta_initial_draw = beta_initial_draw_inp;
}

void
inline
bqreg_t::gibbs(
    const size_t n_burnin_draws, 
    const size_t n_keep_draws,
    const size_t thinning_factor,
    Mat_t& beta_draws, 
    Mat_t& z_draws, 
    ColVec_t& sigma_draws
)
{
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
}

#endif
