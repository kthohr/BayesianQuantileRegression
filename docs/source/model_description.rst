.. Copyright (c) 2021-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Model Description
=================

Classical Quantile Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define the quantile function of a random variable, :math:`Y`, conditional on a random vector, :math:`X`, as:

.. math::

    Q_{Y | X} (\tau) := \inf \left\{ y : F_{Y|X}(y) \geq \tau \right\}

where :math:`\tau \in [0,1]` and :math:`F_{Y|X}` denotes the conditional distribution function of :math:`Y` given :math:`X`.

Quantile regression models the conditional quantile function of a continuous random variable as an affine function of a feature set, :math:`X`, and a parameter vector, 
:math:`\beta`. That is: :math:`Q_{Y | X} (\tau) = X \beta(\tau)`, where we emphasize that the parameter vector, :math:`\beta`, varies by the quantile of interest.

The standard frequentist estimator for :math:`\beta(\tau)` is given by the solution to the following optimization problem:

.. math::

    \hat{\beta}(\tau) = \arg \min_{\beta} \left\{ \sum_{i=1}^N \rho_{\tau} (Y_i - X_i \beta) \right\}

where :math:`\rho_{\tau}` denotes the check function:

.. math::

    \rho_{\tau} (y) = y (\tau - \mathbf{1}[y < 0])

In practice, this optimization problem is solved by linear programming techniques.

Bayesian Quantile Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bayesian version of the classical quantile regression problem assumes that the quantile "error" follows an Asymmetric Laplace Distribution (ALD), the density function of which is given by:

.. math::

    f(y | \mu, \sigma, \tau) = \frac{\tau (1 - \tau)}{\sigma} \exp \left( - \rho_{\tau} \left( \frac{y - \mu}{\sigma} \right) \right)

where :math:`\mu` is the location parameter, :math:`\sigma` is the scale parameter, and :math:`\tau` is the "asymmetry" parameter. Note that when :math:`\tau = 0.5`, the ALD density collapses 
to that of the Laplace distribution (with a rescaled scaling parameter).

The statistical model is defined using a mean-variance mixture representation of the ALD:

.. math::

    \begin{aligned}
        Y_i &= X_i \beta (\tau) + \theta(\tau) \sigma z_i + \omega(\tau) \sigma \sqrt{z_i} u_i \\
        u_i &\stackrel{\text{iid}}{\sim} N(0,1) \\
        z_i &\stackrel{\text{iid}}{\sim} E(1)
    \end{aligned}

where the quantile-dependent parameters :math:`\theta(\tau)` and :math:`\omega(\tau)` are defined as:

.. math::

    \begin{aligned}
        \theta(\tau) &= \frac{1 - 2\tau}{\tau (1 - \tau)} \\
        \omega(\tau) &= \sqrt{ \frac{2}{\tau (1 - \tau)} }
    \end{aligned}

The likelihood function is then given by:

.. math::

    L(\beta, \sigma | Y, X, Z, \tau) \propto \prod_{i=1}^N \frac{1}{\omega \sigma \sqrt{z_i}} \exp \left( - \frac{(Y_i - X_i \beta - \theta \sigma z_i )^2}{2 \omega^2 \sigma^2 z_i} \right)


The Gibbs sampling procedure of Kozumi and Kobayashi (2011) proceeds in three blocks.

1. Draw :math:`\beta`:

  .. math::

    \begin{aligned}
        \beta | Y, X, Z, \sigma &\sim N(\widetilde{\beta}, \widetilde{V}) \\
        \widetilde{\beta} &= \widetilde{V} \left( V_0^{-1} \beta_0 + \frac{1}{\omega^2 \sigma^2} \sum_{i=1}^N \frac{1}{z_i} X_i^\top (Y_i - \theta \sigma z_i) \right) \\
        \widetilde{V}^{-1} &= V_0^{-1} + \frac{1}{\omega^2 \sigma^2} \sum_{i=1}^N \frac{1}{z_i} X_i^\top X_i
    \end{aligned}

2. Draw :math:`\{ z_i \}_{i=1}^N`:

  .. math::

    \begin{aligned}
        z_i | Y, X, \beta, \sigma &\sim g(x; a, b) \propto \frac{1}{\sqrt{x}} \exp \left( - \frac{1}{2} (a^2 x + b_i^2 x^{-1}) \right) \\
        a &= \sqrt{ \frac{2}{\sigma^2} + \frac{\theta^2}{\sigma^2 \omega^2} } \\
        b_i &= \frac{|Y_i - X_i \beta|}{\sigma \omega}
    \end{aligned}

  The sampling density for :math:`z_i` is proportional to a generalized inverse-Gaussian distribution.

3. Draw :math:`\sigma`:

  .. math::

    \begin{aligned}
        \sigma | Y, X, Z, \beta &\sim IG(\tilde{\gamma}_0, \tilde{\gamma}_1) \\
        \tilde{\gamma}_0 &= \gamma_0 + \frac{3}{2} N \\
        \tilde{\gamma}_1 &= \gamma_1 + \sum_{i=1}^N \left[ \sigma z_i + \frac{1}{2 \omega^2 \sigma z_i} ( Y_i - X_i \beta - \theta \sigma z_i ) \right]
    \end{aligned}


----
