.. Copyright (c) 2021-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

R
=

Class Declaration
-----------------

TBW


----

Example
--------

Code to run a median regression example is given below.

.. code:: R

    library(BQReg.Rcpp)

    set.seed(1900)

    #

    n <- 1000
    K <- 3

    X <- matrix(rnorm(n*K),n,K)
    X[,1] <- 1


    beta0 <- matrix(runif(K,1,2), ncol = 1)
    beta0[1] <- 5.0

    Y <- X %*% beta0 + rnorm(n)

    #

    bqreg_obj <- new(bqreg)

    bqreg_obj$load_data(Y, X)

    #

    beta_bar <- matrix(0, K, 1)

    Vbar <- diag(K) * 1/0.001

    n0 <- 3
    s0 <- 3

    bqreg_obj$set_prior_params(beta_bar, Vbar, n0, s0)

    #

    beta_hat <- solve( t(X) %*% X, t(X) %*% Y )
    bqreg_obj$set_initial_beta_draw(beta_hat)

    #

    tau <- 0.5

    bqreg_obj$set_quantile_target(tau)

    n_burnin_draws <- 10000
    n_keep_draws <- 10000

    gibbs_res <- bqreg_obj$gibbs(n_burnin_draws, n_keep_draws, 0)

    rowMeans(gibbs_res$beta_draws)


----
