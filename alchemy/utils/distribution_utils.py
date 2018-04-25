# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.contrib import distributions


class CustomScaleMixture(object):
  def __init__(self, pi=.01, sigma1=.25, sigma2=.01):
    self.mu, self.pi, self.sigma1, self.sigma2 = map(
        np.float32, (0.0, pi, sigma1, sigma2))

  def log_prob(self, x):
    n1 = distributions.Normal(self.mu, self.sigma1)
    n2 = distributions.Normal(self.mu, self.sigma2)
    mix1 = math_ops.reduce_sum(n1.log_prob(x), -1) + math_ops.log(self.pi)
    mix2 = math_ops.reduce_sum(n2.log_prob(x), -1) + math_ops.log(np.float32(1.0 - self.pi))
    prior_mix = array_ops.stack([mix1, mix2])
    lse_mix = math_ops.reduce_logsumexp(prior_mix, [0])
    return math_ops.reduce_sum(lse_mix)


def custom_scale_mixture_prior_factory(pi=.01, sigma1=.25, sigma2=.01):
  """Creates a prior builder and a posterior builder from the pi, sigma1 and sigma2.

  Example usage:

    ```
    import sonnet.python.custom_getters.bayes_by_backprop as bbb

    prior_builder, posterior_builder = custom_scale_mixture_prior_factory(
        pi=.01, sigma1=.25, sigma2=.01)

    sampling_mode_ph = tf.placeholder(tf.string, [])
    get_bbb_variable_fn = bbb.bayes_by_backprop_getter(
        posterior_builder=posterior_builder,
        prior_builder=prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=sampling_mode_ph)
    ```
  """
  def custom_scale_mixture_prior_builder(getter, name, *args, **kwargs):
    del getter
    del name
    del args
    del kwargs
    return CustomScaleMixture(pi=pi, sigma1=sigma1, sigma2=sigma2)

  def custom_posterior_builder(getter, name, *args, **kwargs):
    del args
    parameter_shapes = distributions.Normal.param_static_shapes(
        kwargs.get("shape"))

    dtype = kwargs.get("dtype", dtypes.float32)

    # The standard deviation of the scale mixture prior.
    prior_stddev = np.sqrt(
        pi * np.square(sigma1) +
        (1 - pi) * np.square(sigma2))

    loc_var = getter(
        "{}/posterior_loc".format(name),
        shape=parameter_shapes["loc"],
        initializer=kwargs.get("initializer", None),
        dtype=dtype,
        trainable=kwargs.get("trainable", True))
    scale_var = getter(
        "{}/posterior_scale".format(name),
        initializer=random_ops.random_uniform(
            minval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
            maxval=np.log(np.exp(prior_stddev / 1.0) - 1.0),
            dtype=dtype,
            shape=parameter_shapes["scale"]))
    return distributions.Normal(
        loc=loc_var,
        scale=nn_ops.softplus(scale_var) + 1e-5,
        name="{}/posterior_dist".format(name))
  return custom_scale_mixture_prior_builder, custom_posterior_builder


def signal_to_noise_ratio(dist):
  """Signal to noise ratio of a normal distribution, N(loc, softplus(scale)).

  Args:
    dist: `tf.distribution.Normal`.
  """
  return math_ops.abs(dist.loc) / math_ops.log(math_ops.exp(dist.scale) - 1)
