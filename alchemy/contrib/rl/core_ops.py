# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops


def discount_rewards(rewards, sequence_length, max_sequence_length, discount=.95):
  """Compute and return the discounted/filtered reward.

  Arguments:
    rewards: 1D or 2D `tf.Tensor`, contiguous sequence(s) of rewards.
    sequence_length: 0D or 1D `tf.Tensor`, the length of the `rewards` sequence(s).
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    discount: 0D scalar `tf.Tensor`, the discount factor (gamma).

  Returns:
    Tensor with the same shape as `rewards`.
  """
  discount = ops.convert_to_tensor(discount, dtype=rewards.dtype)
  discounted_reward = tensor_array_ops.TensorArray(
      size=max_sequence_length, dtype=rewards.dtype)
  running_reward = constant_op.constant(0., dtype=rewards.dtype, name='running_reward')

  def body_fn(t, rr, r, dr):
    rr = discount * rr + r[t]
    dr = dr.write(t, rr)
    return (t - 1, rr, r, dr)

  def condition_fn(t, rr, r, dr):
    return t > -1

  _, _, _, discounted_reward = control_flow_ops.while_loop(
      condition_fn, body_fn,
      [sequence_length - 1, running_reward, rewards, discounted_reward])

  discounted_reward = discounted_reward.stack()
  return discounted_reward
