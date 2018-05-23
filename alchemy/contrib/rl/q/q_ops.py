# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from alchemy.utils import assert_utils
from alchemy.utils import sequence_utils
from alchemy.contrib.rl import core_ops


def gather_along_second_axis(data, indices):
  """Super-weird way to select by a dimension.
  This can be refactored into a single call with an axis argument.
  """
  ndims = len(data.get_shape().as_list())
  shape = array_ops.shape(data)
  re_shape = [shape[0] * shape[1]]
  indices = array_ops.reshape(indices, re_shape)
  for idx in range(2, ndims):
    re_shape.append(shape[idx])
  data = array_ops.reshape(data, re_shape)
  batch_offset = math_ops.range(0, array_ops.shape(data)[0])
  flat_indices = array_ops.stack([batch_offset, indices], axis=1)
  two_d = gen_array_ops.gather_nd(data, flat_indices)
  three_d = gen_array_ops.reshape(two_d, [shape[0], shape[1], -1])
  return three_d


def expected_q_value(reward, action, action_value, next_action_value, max_sequence_length, weights=1., discount=.95):
  """Computes the expected q returns and values.

  This covers architectures such as DQN, Double-DQN, Dueling-DQN and Noisy-DQN.

  Arguments:
    reward: 1D or 2D `tf.Tensor`, contiguous sequence(s) of rewards.
    action: 1D or 2D `tf.Tensor`, contiguous sequence(s) of actions.
    next_action_value: `tf.Tensor`, `list` or `tuple` of 2 `tf.Tensor`s, where the first entry is
        the `model(next_state) = action_value`, and the second is `target(next_state) = action_value`
    max_sequence_length: `int` or `list`, maximum length(s) of rewards.
    weights: `tf.Tensor`, the weights/mask to apply to the result.
    discount: 0D scalar, the discount factor (gamma).

  Returns:
    `tuple` containing the `q_value` `tf.Tensor` and `expected_q_value` `tf.Tensor`.

  Reference:
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
  """
  weights = ops.convert_to_tensor(weights, dtype=reward.dtype)
  discount = ops.convert_to_tensor(discount, dtype=reward.dtype)

  lda = action_value.get_shape()[-1].value
  q_value = gather_along_second_axis(action_value, action)
  q_value.set_shape([None, max_sequence_length, lda])

  if isinstance(next_action_value, tuple) or isinstance(next_action_value, list):
    assert_utils.assert_true(
        len(next_action_value) == 2,
        '`next_action_value` must be a `tuple` of length = 2')
    next_action_value, target_next_action_value = next_action_value
    lda = next_action_value.get_shape()[-1].value
    next_q_value = gather_along_second_axis(
        next_action_value,
        math_ops.argmax(target_next_action_value, -1, output_type=dtypes.int32))
    next_q_value.set_shape([None, max_sequence_length, lda])
  else:
    lda = next_action_value.get_shape()[-1].value
    next_q_value = gather_along_second_axis(
        next_action_value,
        math_ops.argmax(next_action_value, -1, output_type=dtypes.int32))
    next_q_value.set_shape([None, max_sequence_length, lda])

  expected_q_value = array_ops.expand_dims(
      reward, -1) + discount * next_q_value * array_ops.expand_dims(
          weights, -1)
  return (q_value, expected_q_value)


# WARNING: This doesn't work (I think)
# TODO(wenkesj): figure out what is wrong with this.
def q_quantile(q_dist, expected_q_dist):
  shape = array_ops.shape(q_dist)
  batch_size, sequence_size = shape[0], shape[1]
  num_quantiles = q_dist.get_shape()[-1].value

  big_expected_q_dist = array_ops.transpose(
      gen_array_ops.reshape(
          array_ops.tile(
              expected_q_dist, [1, 1, num_quantiles]),
          [batch_size, sequence_size, num_quantiles, num_quantiles]),
      perm=[0, 1, 3, 2])

  big_q_dist = gen_array_ops.reshape(
      array_ops.tile(
          q_dist,
          [1, 1, num_quantiles]),
      [batch_size, sequence_size, num_quantiles, num_quantiles])
  return (big_q_dist, big_expected_q_dist)
