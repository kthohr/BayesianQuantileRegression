# import libraries
import numpy as np
from pybqreg import BayesianQuantileRegression

# generate data
n = 500
K = 3

X = np.random.normal(0.0, 1.0, [n, K])
X[:,0] = np.ones(n)

beta0 = np.random.uniform(1.0, 2.0, K)
beta0[0] = 5.0

Y = np.matmul(X, beta0) + np.random.normal(0.0, 1.0, n)

# initialize object
obj = BayesianQuantileRegression(Y,X)

# set prior pars
beta_bar = np.zeros(K)

Vbar = np.zeros([K, K])
np.fill_diagonal(Vbar, 1.0/0.001)

n0 = 3
s0 = 3

obj.set_prior_params(beta_bar, Vbar, n0, s0)

# (optional) set the initial draw for beta
beta_hat = np.linalg.solve( np.matmul(X.transpose(),X), np.matmul(X.transpose(),Y) )
obj.set_initial_beta_draw(beta_hat)

# (optional) set the number of OpenMP threads
obj.set_omp_n_threads(4)
obj.get_omp_n_threads()


# set the target quantile (tau) and run the Gibbs sampler
tau = 0.5
n_burnin_draws = 10000
n_keep_draws = 10000

beta_draws, nu_draws, sigma_draws = obj.fit(tau, n_burnin_draws, n_keep_draws, 0)

np.mean(beta_draws, axis = 1)

np.mean(sigma_draws)

np.mean(nu_draws, axis = 1)
