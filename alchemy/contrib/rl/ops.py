# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops


def discount_rewards(reward, sequence_length, max_sequence_length, discount=.9):
  """Compute and return the discounted/filtered reward."""
  discounted_reward = tensor_array_ops.TensorArray(
      size=max_sequence_length, dtype=reward.dtype)
  running_reward = constant_op.constant(0., dtype=reward.dtype, name='running_reward')

  def body_fn(t, rr, r, dr):
    rr = discount * rr + r[t]
    dr = dr.write(t, rr)
    return (t - 1, rr, r, dr)

  def condition_fn(t, rr, r, dr):
    return t > -1

  _, _, _, discounted_reward = control_flow_ops.while_loop(
      condition_fn, body_fn,
      [sequence_length - 1, running_reward, reward, discounted_reward])

  discounted_reward = discounted_reward.stack()
  return discounted_reward
