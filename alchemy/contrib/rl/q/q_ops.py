# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from alchemy.utils import assert_utils
from alchemy.utils import sequence_utils
from alchemy.contrib.rl import core_ops


def slow_gather(params, indices):
  mask = array_ops.one_hot(indices, array_ops.shape(params)[-1])
  return math_ops.reduce_sum(mask * params, -1)


def expected_q_value(reward, action, action_value, next_action_value, weights=1., discount=.95):
  """Computes the expected q returns and values.

  This covers architectures such as DQN, Double-DQN, Dueling-DQN and Noisy-DQN.

  Arguments:
    rewards: 1D or 2D `tf.Tensor`, contiguous sequence(s) of rewards.
    action: 1D or 2D `tf.Tensor`, contiguous sequence(s) of actions.
    next_action_value: `tf.Tensor`, `list` or `tuple` of 2 `tf.Tensor`s, where the first entry is
        the `model(next_state) = action_value`, and the second is `target(next_state) = action_value`
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).

  Returns:
    `tuple` containing the `q_value` `tf.Tensor` and `expected_q_value` `tf.Tensor`.

  Reference:
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
  """
  weights = ops.convert_to_tensor(weights, dtype=reward.dtype)
  discount = ops.convert_to_tensor(discount, dtype=reward.dtype)

  q_value = slow_gather(action_value, action)

  if isinstance(next_action_value, tuple) or isinstance(next_action_value, list):
    assert_utils.assert_true(
        len(next_action_value) == 2,
        '`next_action_value` must be a `tuple` of length = 2')
    next_action_value, target_next_action_value = next_action_value
    next_q_value = slow_gather(
        target_next_action_value,
        math_ops.argmax(next_action_value, -1))
  else:
    next_q_value = slow_gather(
        next_action_value,
        math_ops.argmax(next_action_value, -1))

  expected_q_value = reward + discount * next_q_value * weights
  return (q_value, expected_q_value)


def q_quantile(q_dist):
  shape = array_ops.shape(q_dist)
  batch_size, sequence_size = shape[0], shape[1]
  k = q_dist.get_shape()[-1].value
  _, quantile_idx = nn_ops.top_k(q_dist, k=k) # sort quantiles in ascending order
  quantile_idx = array_ops.reverse(quantile_idx, [-1])

  # midpoint quantile targets
  tau_hat = math_ops.linspace(0.0, 1.0 - 1. / k, k) + .5 / k
  tau_hat = sequence_utils.expand_dims(tau_hat, axes=[0, 1])
  tau_hat = array_ops.tile(tau_hat, [batch_size, sequence_size, 1])

  # I've got 51 cumulative prob(lem)s ;)
  tau_op = slow_gather(
      array_ops.expand_dims(tau_hat, -1), quantile_idx)
  return tau_op
