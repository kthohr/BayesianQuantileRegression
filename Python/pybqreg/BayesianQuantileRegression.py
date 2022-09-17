################################################################################
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
################################################################################

from typing import Union
import numpy as np
import pandas as pd

from bqreg_wrapper import bqreg

class BayesianQuantileRegression:
    '''
    Bayesian Quantile Regression class
    '''
    def __init__(
        self,
        target: Union[pd.Series, pd.DataFrame, np.ndarray],
        features: Union[pd.Series, pd.DataFrame, np.ndarray]
    ):
        '''
        Initialize the BayesianQuantileRegression class

            Parameters:
                target: An n x 1 vector defining the target variable (Y)
                features: An n x K matrix of features (X)
        '''

        self.n = target.shape[0]

        if len(features.shape) == 1:
            self.K = 1
        else:
            self.K = features.shape[1]

        if isinstance(target, (pd.Series, pd.DataFrame)):
            self.Y = target.to_numpy()
        elif isinstance(target, np.ndarray):
            self.Y = target
        else:
            raise Exception("The 'target' vector must be of type 'pandas.DataFrame', 'pandas.Series', or 'numpy.ndarray'")
        
        if isinstance(features, (pd.Series, pd.DataFrame)):
            self.X = features.to_numpy()
        elif isinstance(features, np.ndarray):
            self.X = features
        else:
            raise Exception("The 'features' vector/matrix must be of type 'pandas.DataFrame', 'pandas.Series', or 'numpy.ndarray'")

        if self.K == 1:
            self.X = self.X[:, np.newaxis]

        self.bqreg_obj = bqreg()

        self.bqreg_obj.load_data(self.Y, self.X)

        self.bqreg_obj.set_prior_params(np.zeros(self.K), np.eye(self.K), 3, 3)
        self.bqreg_obj.set_initial_beta_draw(np.zeros(self.K))
    
    def get_omp_n_threads(
        self
    ) -> int:
        '''
        Get the current value determining the number of OpenMP threads to use with the Gibbs sampler

            Returns:
                An integer value
        '''
        return self.bqreg_obj.get_omp_n_threads()
    
    def set_omp_n_threads(
        self,
        omp_n_threads: int
    ):
        '''
        Set the number of OpenMP threads to use for the Gibbs sampler

            Parameters:
                omp_n_threads: An integer value
        '''
        self.bqreg_obj.set_omp_n_threads(omp_n_threads)
    
    def set_prior_params(
        self,
        beta_mean: np.ndarray, 
        beta_var: np.ndarray,
        sigma_shape: float,
        sigma_scale: float
    ):
        '''
        Set the parameters of the prior distributions

            Parameters:
                beta_mean: Mean vector of the normal prior for beta
                beta_var: Variance matrix of the normal prior for beta
                sigma_shape: Shape parameter of the prior distribution for sigma^2
                sigma_scale: Scale parameter of the prior distribution for sigma^2
        '''
        self.bqreg_obj.set_prior_params(beta_mean, beta_var, sigma_shape, sigma_scale)

    def set_initial_beta_draw(
        self,
        beta_initial_draw: np.ndarray
    ):
        '''
        Set the initial values of beta for the Gibbs sampler

            Parameters:
                beta_initial_draw: Initial draw vector
        '''
        self.bqreg_obj.set_initial_beta_draw(beta_initial_draw)

    def fit(
        self,
        tau: float = 0.5,
        n_burnin_draws: int = 1000,
        n_keep_draws: int = 1000
    ) -> tuple:
        '''
        Fit method for the BayesianQuantileRegression class

            Parameters:
                tau: the target quantile value
                n_burnin_draws: the number of burn-in draws
                n_keep_draws: the number of post-burn-in draws
            
            Returns:
                A tuple of matrices containing posterior draws, ordered as follows: (beta, nu, sigma)
        '''
        
        self.bqreg_obj.set_quantile_target(tau)

        draws = self.bqreg_obj.gibbs(n_burnin_draws, n_keep_draws)

        return draws[0], draws[1], draws[2] # (beta, nu, sigma)
