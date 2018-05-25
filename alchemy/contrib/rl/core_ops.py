# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope

from alchemy.utils import shortcuts


def discount(rewards, max_sequence_length, initial_value=None, weights=1., discount=.95, time_major=False):
  """Compute and return the discounted/filtered reward.

  Arguments:
    rewards: 1D or 2D `tf.Tensor`, contiguous sequence(s) of rewards.
    max_sequence_length: `int`, maximum length(s) of rewards tensor.
    initial_value: optional `tf.Tensor`, optional initial discounted value.
    weights: optional `tf.Tensor` the weights/mask to apply to the result.
    discount: 0D scalar `tf.Tensor`, the discount factor (gamma).
    time_major: `Boolean`, if rewards is 2D and already time_major, i.e. [time, batch_size].

  Raises:
    `ValueError` when reward is not a 1D or 2D `tf.Tensor`.

  Returns:
    Tensor with the same shape as `rewards`.
  """
  has_initial_value = isinstance(initial_value, ops.Tensor)

  weights = ops.convert_to_tensor(weights, dtype=rewards.dtype)
  discount = ops.convert_to_tensor(discount, dtype=rewards.dtype)

  dims = shortcuts.ndims(rewards)
  shape = array_ops.shape(rewards)

  if dims == 2:
    gather = lambda x, x_t: x[x_t, :]
    if time_major:
      element_shape = [shape[-1]]
    else:
      element_shape = [shape[0]]
  elif dims == 1:
    gather = lambda x, x_t: x[x_t]
    element_shape = []
  else:
    raise ValueError('reward must be a 1D or 2D tensor, got {}'.format(dims))

  if has_initial_value:
    if not time_major:
      initial_value = array_ops.transpose(initial_value)
    running_reward = initial_value
    def body_fn(t, rr, r, dr):
      rr = discount * rr + array_ops.expand_dims(gather(r, t), 0)
      dr = dr.write(t, gather(rr, t))
      return (t - 1, rr, r, dr)
  else:
    running_reward = array_ops.zeros(
            element_shape, dtype=rewards.dtype, name='running_reward')
    def body_fn(t, rr, r, dr):
      rr = discount * rr + gather(r, t)
      dr = dr.write(t, rr)
      return (t - 1, rr, r, dr)

  discounted_reward = tensor_array_ops.TensorArray(
      rewards.dtype,
      size=max_sequence_length,
      name='discounted_reward')

  def condition_fn(t, rr, r, dr):
    return t > -1

  _, _, _, discounted_reward = control_flow_ops.while_loop(
      condition_fn, body_fn,
      [max_sequence_length - 1,
       running_reward,
       rewards if time_major else array_ops.transpose(rewards),
       discounted_reward])

  discounted_reward = discounted_reward.stack()
  if not time_major:
    discounted_reward = array_ops.transpose(discounted_reward)
  return discounted_reward * weights
