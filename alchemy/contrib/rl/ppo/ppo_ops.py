# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import normalization

from alchemy.utils import assert_utils
from alchemy.utils import distribution_utils
from alchemy.utils import shortcuts
from alchemy.utils import sequence_utils
from alchemy.contrib.rl import core_ops


def generalized_advantage_estimate(rewards,
                                   values,
                                   sequence_length,
                                   max_sequence_length,
                                   weights=1.,
                                   discount=.9,
                                   lambda_td=.95,
                                   normalize_advantages=False,
                                   time_major=False):
  """Computes the GAE algorithm.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    values: 1D or 2D Tensor, contiguous sequence(s) of value estimates.
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).
    lambda_td: 0D scalar, the td(lambda) factor (lambda).
    normalize_advantages: `Boolean`, if the advantages should be normalized.
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].

  Returns:
    `tuple` of Tensors with the same shape as `rewards`: (advantages, returns).
  """
  lambda_td = ops.convert_to_tensor(lambda_td, dtype=rewards.dtype)
  next_values = sequence_utils.shift_right(values)
  delta_t = rewards + discount * next_values - values

  advantage_op = discounted_op = core_ops.discount_rewards(
      delta_t,
      max_sequence_length=max_sequence_length,
      weights=weights,
      discount=discount * lambda_td,
      time_major=time_major)

  advantage_op.set_shape([None, max_sequence_length])
  returns = advantage_op + values

  return standardize(advantage_op, sequence_length, max_sequence_length), returns


def standardize(op, length, max_length):
  mask = math_ops.cast(array_ops.sequence_mask(length, maxlen=max_length), op.dtype)

  length_expanded = array_ops.expand_dims(length, -1)
  mean_op = math_ops.cumsum(
      op, axis=-1, reverse=False) / math_ops.cast(
          length_expanded, op.dtype)
  mean_op *= mask

  diff_op = op - mean_op
  stdv_op = math_ops.sqrt(
      math_ops.cumsum(
          diff_op, axis=-1, reverse=False) / math_ops.cast(
              length_expanded, op.dtype))
  stdv_op *= mask
  return ((diff_op + distribution_utils.epsilon) / (
      stdv_op + distribution_utils.epsilon)) * mask
