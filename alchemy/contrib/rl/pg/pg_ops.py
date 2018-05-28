# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from alchemy.utils import sequence_utils
from alchemy.utils import shortcuts
from alchemy.contrib.rl import core_ops


def advantage(rewards, sequence_length, max_sequence_length,
              weights=1.,
              discount=.95,
              time_major=False,
              scale=False,
              center=False):
  """Compute the advantage based on the baseline discounted rewards.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D Tensor, the length of the `rewards` sequence(s).
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].
    scale: `Boolean`, if discounts are scaled by the aggregate baseline.
    center: `Boolean`, if discounts are centered by the aggregate baseline.

  Returns:
    Tensor with the same shape as `rewards`.
  """
  discounted_reward_op = core_ops.discount(
      rewards,
      max_sequence_length=max_sequence_length,
      weights=weights,
      discount=discount,
      time_major=time_major)
  discounted_reward_op.set_shape([None, max_sequence_length])

  if scale or center:
    discounted_reward_op = shortcuts.cumnormalize(
        discounted_reward_op, sequence_length, max_sequence_length,
        scale=scale, center=center)
  return discounted_reward_op


def generalized_advantage_estimate(rewards,
                                   values,
                                   sequence_length,
                                   max_sequence_length,
                                   weights=1.,
                                   discount=.9,
                                   lambda_td=.95,
                                   time_major=False,
                                   scale=False,
                                   center=False):
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
    scale: `Boolean`, if discounts are scaled by the aggregate baseline.
    center: `Boolean`, if discounts are centered by the aggregate baseline.

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
  next_values = sequence_utils.shift(values, axis=-1, rotations=-1)
  delta = (rewards + discount * next_values - values) * weights

  advantage_op = core_ops.discount(
      delta,
      max_sequence_length=max_sequence_length,
      weights=weights,
      discount=discount * lambda_td,
      time_major=time_major)

  returns_op = advantage_op + values
  returns_op.set_shape([None, max_sequence_length])

  if scale or center:
    advantage_op = shortcuts.cumnormalize(
        advantage_op, sequence_length, max_sequence_length,
        scale=scale, center=center)
  return advantage_op, returns_op
