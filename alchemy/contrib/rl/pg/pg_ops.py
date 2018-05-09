# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from alchemy.contrib.rl import core_ops


def advantage(rewards, sequence_length, max_sequence_length, discount=.95, time_major=False):
  """Compute the advantage based on the baseline discounted rewards.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D Tensor, the length of the `rewards` sequence(s).
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    discount: 0D scalar, the discount factor (gamma).
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].

  Returns:
    Tensor with the same shape as `rewards`.
  """
  discounted_reward_op = core_ops.discount_rewards(
      rewards,
      max_sequence_length=max_sequence_length,
      discount=discount,
      time_major=time_major)

  baseline_op = math_ops.cumsum(
      discounted_reward_op,
      axis=-1,
      reverse=False) / math_ops.cast(
          array_ops.expand_dims(sequence_length, -1),
          discounted_reward_op.dtype)

  baseline_op *= math_ops.cast(
      array_ops.sequence_mask(sequence_length), baseline_op.dtype)
  return discounted_reward_op - baseline_op, discounted_reward_op, baseline_op
