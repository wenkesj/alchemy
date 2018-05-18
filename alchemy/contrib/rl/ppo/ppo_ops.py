# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from alchemy.utils import shortcuts
from alchemy.utils import sequence_utils
from alchemy.utils import distribution_utils
from alchemy.contrib.rl import core_ops


def generalized_advantage_estimate(rewards,
                                   values,
                                   sequence_length,
                                   max_sequence_length,
                                   weights=1.,
                                   discount=.9,
                                   lambda_td=.95,
                                   time_major=False,
                                   normalize_advantages=True):
  """Computes the GAE algorithm.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D Tensor, the length of the `rewards` sequence(s).
    values: 1D or 2D Tensor, contiguous sequence(s) of value estimates.
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).
    lambda_td: 0D scalar, the td(lambda) factor (lambda).
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].

  Returns:
    `tuple` of Tensors with the same shape as `rewards`: (advantages, returns).
  """
  discount = ops.convert_to_tensor(discount, dtype=rewards.dtype)
  lambda_td = ops.convert_to_tensor(lambda_td, dtype=rewards.dtype)

  mask = math_ops.cast(
      array_ops.sequence_mask(
          sequence_length, maxlen=max_sequence_length),
      rewards.dtype)
  batch_size = array_ops.shape(values)[0]
  next_values = array_ops.concat(
      [values[:, 1:], array_ops.zeros([batch_size, 1])],
      axis=-1)
  delta = (rewards + discount * next_values - values) * weights

  advantage_op = core_ops.discount_rewards(
      delta,
      max_sequence_length=max_sequence_length,
      weights=weights,
      discount=discount * lambda_td,
      time_major=time_major)

  returns_op = advantage_op + values
  returns_op.set_shape([None, max_sequence_length])

  if normalize_advantages:
    advantage_op = shortcuts.batch_norm(advantage_op)
  return advantage_op, returns_op
