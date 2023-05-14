# Bayesian Quantile Regression

A light-weight C++ implementation of Bayesian quantile regression using the asymmetric Laplace density representation of the problem, with wrappers for Python and R.

# Installation

``` bash
# clone optim into the current directory
git clone https://github.com/kthohr/BayesianQuantileRegression ./BayesianQuantileRegression

# change directory
cd ./BayesianQuantileRegression

# clone necessary submodules
git submodule update --init
```

## Python wrapper

If you do not already have Eigen3 and Pybind11, clone and set environment variables:

```bash
# close Eigen3
git clone https://gitlab.com/libeigen/eigen.git

# close Pybind11
git clone https://github.com/pybind/pybind11.git

# set environment variables
export EIGEN_INCLUDE_PATH = "<path to this directory>/eigen"
export PYBIND11_INCLUDE_PATH = "<path to this directory>/pybind11/include"
```

```bash
# change directory into the Python subdirectory
cd BayesianQuantileRegression/Python

# install the package
python3 -m pip install . --user
```

## R wrapper

```bash
# change directory into the R subdirectory
cd BayesianQuantileRegression/R

# install the package
R CMD INSTALL .
```
