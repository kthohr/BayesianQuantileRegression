# Bayesian Quantile Regression

A light-weight C++ implementation of Bayesian quantile regression using the asymmetric Laplace density representation of the problem.

# Installation

``` bash
# clone optim into the current directory
git clone https://github.com/kthohr/BayesianQuantileRegression ./BayesianQuantileRegression

# change directory
cd ./BayesianQuantileRegression

# clone necessary submodules
git submodule update --init
```

## Python

```bash
# change directory into the Python subdirectory
cd Python

python3 -m pip install . --user
```

## R

```bash
# change directory into the R subdirectory
cd R

R CMD INSTALL .
```
