# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import math_ops

from alchemy.contrib.rl import core_ops


def advantage(rewards, sequence_length, max_sequence_length, discount=.95):
  """Compute the advantage based on the baseline discounted rewards.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D Tensor, the length of the `rewards` sequence(s).
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    discount: 0D scalar, the discount factor (gamma).

  Returns:
    Tensor with the same shape as `rewards`.
  """
  discounted_reward_op = core_ops.discount_rewards(
      rewards,
      sequence_length=sequence_length,
      max_sequence_length=max_sequence_length,
      discount=discount)

  baseline_op = math_ops.cumsum(discounted_reward_op, reverse=False) / math_ops.cast(
      sequence_length, discounted_reward_op.dtype)
  return discounted_reward_op - baseline_op
