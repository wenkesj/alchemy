# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import beta

from alchemy.utils import distribution_utils


class BetaMinMax(beta.Beta):

  """
  A Beta distribution such that samples/probs are bounded by [`min_value`, `max_value`].
  """

  def __init__(self,
               concentration1=None,
               concentration0=None,
               min_value=None,
               max_value=None,
               validate_args=False,
               allow_nan_stats=True,
               name='BetaMinMax'):

    super(BetaMinMax, self).__init__(
        concentration1=concentration1,
        concentration0=concentration0,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self.min_value = ops.convert_to_tensor(min_value, dtype=self.dtype)
    self.max_value = ops.convert_to_tensor(max_value, dtype=self.dtype)

  def sample(self, sample_shape=(), seed=None, name='sample'):
    return self.min_value + (self.max_value - self.min_value) * super(
        BetaMinMax, self).sample(
            sample_shape=sample_shape,
            seed=seed,
            name=name)

  def log_prob(self, value, name='log_prob'):
    value = (value - self.min_value) / (self.max_value - self.min_value)
    value = math_ops.minimum(value, (1.0 - distribution_utils.epsilon))
    return super(BetaMinMax, self).log_prob(value, name=name)
