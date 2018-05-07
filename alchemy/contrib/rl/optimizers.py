# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import adam
from tensorflow.python.training import optimizer


class PolicyGradOptimizer(optimizer.Optimizer):
  """
  Implements the policy gradient method.
  """
  def __init__(self, opt, discounts,
               use_locking=False,
               name="PolicyGradOptimizer"):
    super(PolicyGradOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    self._discounts = ops.convert_to_tensor(discounts, dtype=dtypes.float32)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    discounted_grads_and_vars = [(g * self._discounts, v) for g, v in grads_and_vars]
    return self._opt.apply_gradients(discounted_grads_and_vars, *args, **kwargs)
