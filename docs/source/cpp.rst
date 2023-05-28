.. Copyright (c) 2021-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

C++
===

Class Declaration
-----------------

.. _bqreg-class-ref1:
.. doxygenclass:: bqreg_t
   :project: bqreg
   :members:


----

Example
--------

Code to run a median regression example is given below.

.. code:: cpp

    #include "bqreg.hpp"

    inline
    Eigen::MatrixXd
    eigen_randn(const size_t nr, const size_t nc)
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return Eigen::MatrixXd{ nr, nc }.unaryExpr([&](double x) { (void)(x); return dist(gen); });
    }

    int main()
    {
        // generate data
        const int n = 500;
        const int K = 3;
    
        Eigen::MatrixXd X = eigen_randn(n, K);
        X.col(0).setConstant(1);

        Eigen::VectorXd beta0(3); 
        beta0 << 5.0, 1.3, 1.8;
    
        Eigen::VectorXd Y = (X * beta0).array() + eigen_randn(n, 1).array();
    
        // initialize object
        bqreg::bqreg_t obj = bqreg::bqreg_t(Y, X);

        // set prior pars
        Eigen::VectorXd beta_bar = Eigen::VectorXd::Zero(K);

        Eigen::MatrixXd V0 = Eigen::MatrixXd::Zero(K, K);
        V0.diagonal().array() = 1.0 / 0.001;

        int prior_shape = 3;
        int prior_scale = 3;

        obj.set_prior_params(beta_bar, V0, prior_shape, prior_scale);

        // (optional) set the number of OpenMP threads
        obj.set_omp_n_threads(4);
        std::cout << "Number of OpenMP threads: " << obj.get_omp_n_threads() << std::endl;

        // (optional) set the seed value of the Gibbs sampler RNG
        obj.set_seed_value(1111);

        // set the target quantile
        obj.set_quantile_target(0.5);

        // run Gibbs sampler
        size_t n_burnin_draws = 10000;
        size_t n_keep_draws = 10000;
        size_t thinning_factor = 0;

        Eigen::MatrixXd beta_draws; 
        Eigen::MatrixXd z_draws;
        Eigen::VectorXd sigma_draws;
        
        obj.gibbs(n_burnin_draws, n_keep_draws, thinning_factor, beta_draws, z_draws, sigma_draws);

        //
    
        std::cout << "\n - beta posterior mean:\n" << beta_draws.rowwise().mean() << std::endl;
        
        //
    
        return 0;
    }


----
