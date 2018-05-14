# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops

from alchemy.utils import assert_utils
from alchemy.utils import distribution_utils


def epsilon_greedy(dist, epsilon, deterministic):
  """Compute the mode of the distribution if epsilon < X ~ U(0, 1), else sample.

  Arguments:
    dist: a `tf.distribution.Distribution` to sample/mode from.
    epsilon: scalar `tf.Tensor`.
    deterministic: `Boolean` or `tf.Tensor` boolean, if `True` the mode is always be chosen.

  Raises:
    `AssertionError` if dist is not a `tf.distribution.Distribution`.

  Returns:
    `tf.Tensor` shape of `dist.event_shape`.
  """
  assert_utils.assert_true(
      distribution_utils.is_distribution(dist),
      '`dist must be a `tf.distribution.Distribution.`')

  deterministic_sample = lambda: dist.mode()
  return control_flow_ops.cond(
      deterministic,
      deterministic_sample,
      lambda: control_flow_ops.cond(
          epsilon < random_ops.random_uniform([]),
          deterministic_sample,
          lambda: dist.sample()))
