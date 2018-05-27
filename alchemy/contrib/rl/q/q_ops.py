# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from alchemy.utils import assert_utils
from alchemy.utils import sequence_utils
from alchemy.contrib.rl import core_ops



def expected_q_value(reward, action, action_value, next_action_value,
                     sequence_length, max_sequence_length,
                     weights=1., discount=.95, n_step=False):
  """Computes the expected q returns and values.

  This covers architectures such as DQN, Double-DQN, Dueling-DQN and Noisy-DQN.

  Arguments:
    reward: 1D or 2D `tf.Tensor`, contiguous sequence(s) of rewards.
    action: 1D or 2D `tf.Tensor`, contiguous sequence(s) of actions.
    next_action_value: `tf.Tensor`, `list` or `tuple` of 2 `tf.Tensor`s, where the first entry is
        the `model(next_state) = action_value`, and the second is `target(next_state) = action_value`
    sequence_length: 1D `tf.Tensor`, tensor containing lengths of rewards, action_values, etc..
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).
    n_step: 0D bool, if n-step algorithm should be used, MC sampling with discounts.

  Returns:
    `tuple` containing the `q_value` `tf.Tensor` and `expected_q_value` `tf.Tensor`.

  Reference:
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
  """
  weights = ops.convert_to_tensor(weights, dtype=reward.dtype)
  discount = ops.convert_to_tensor(discount, dtype=reward.dtype)
  n_step = ops.convert_to_tensor(n_step, dtype=dtypes.bool)
  ndim = len(action_value.shape)

  q_value = sequence_utils.gather_along_second_axis(action_value, action)
  if ndim == 4:
    q_value.set_shape([None, max_sequence_length, action_value.shape[-1]])
  elif ndim == 3:
    q_value.set_shape([None, max_sequence_length])

  if isinstance(next_action_value, tuple) or isinstance(next_action_value, list):
    assert_utils.assert_true(
        len(next_action_value) == 2,
        '`next_action_value` must be a `tuple` of length = 2')
    next_action_value, target_next_action_value = next_action_value
    next_q_value = sequence_utils.gather_along_second_axis(
        next_action_value,
        math_ops.argmax(target_next_action_value, -1, output_type=dtypes.int32))
  else:
    next_q_value = sequence_utils.gather_along_second_axis(
        next_action_value,
        math_ops.argmax(next_action_value, -1, output_type=dtypes.int32))

  if ndim == 4:
    next_q_value.set_shape([None, max_sequence_length, action_value.shape[-1]])
  elif ndim == 3:
    next_q_value.set_shape([None, max_sequence_length])

  def true_fn():
    reward_t = core_ops.discount(
        reward,
        max_sequence_length=max_sequence_length,
        # initial_value=next_q_value,
        weights=weights,
        discount=discount)
    discount_t = array_ops.tile(
        array_ops.expand_dims(discount, -1),
        [array_ops.shape(sequence_length)[0]]) ** math_ops.cast(
            sequence_length, dtypes.float32)
    discount_t = array_ops.expand_dims(discount_t, -1)
    if ndim == 4:
      reward_t = array_ops.expand_dims(reward_t, -1)
      discount_t = array_ops.expand_dims(discount_t, -1)
    return reward_t + discount_t * next_q_value

  def false_fn():
    reward_t = reward
    if ndim == 4:
      reward_t = array_ops.expand_dims(reward_t, -1)
    return reward_t + discount * next_q_value

  expected_q_value = control_flow_ops.cond(n_step, true_fn, false_fn)

  if ndim == 4:
    weights = array_ops.expand_dims(weights, -1)
    expected_q_value.set_shape([None, max_sequence_length, action_value.shape[-1]])
  elif ndim == 3:
    expected_q_value.set_shape([None, max_sequence_length])

  return (q_value * weights, expected_q_value * weights)
