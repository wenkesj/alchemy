# -*- coding: utf-8 -*-
from __future__ import absolute_import

from alchemy.utils import distribution_utils
from alchemy.utils import shortcuts
from alchemy.contrib.rl import core_ops


def advantage(rewards, sequence_length, max_sequence_length,
              weights=1.,
              discount=.95,
              time_major=False,
              normalize_advantages=True):
  """Compute the advantage based on the baseline discounted rewards.

  Arguments:
    rewards: 1D or 2D Tensor, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D Tensor, the length of the `rewards` sequence(s).
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].

  Returns:
    Tensor with the same shape as `rewards`.
  """
  discounted_reward_op = core_ops.discount_rewards(
      rewards,
      max_sequence_length=max_sequence_length,
      weights=weights,
      discount=discount,
      time_major=time_major)
  discounted_reward_op.set_shape([None, max_sequence_length])

  aggregate_baseline = shortcuts.cummean(
      discounted_reward_op, sequence_length, max_sequence_length)

  discounted_reward_op = discounted_reward_op - aggregate_baseline

  if normalize_advantages:
    return shortcuts.batch_norm(discounted_reward_op)
  return discounted_reward_op
