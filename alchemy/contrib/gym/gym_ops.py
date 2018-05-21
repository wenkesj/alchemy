# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import math

from gym.core import Space
from gym.spaces import box
from gym.spaces import discrete
from gym.spaces import dict_space
from gym.spaces import multi_discrete
from gym.spaces import multi_binary
from gym.spaces import tuple_space

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.layers import core

from alchemy.utils import assert_utils
from alchemy.utils import array_utils
from alchemy.utils import distribution_utils
from alchemy.contrib.distributions import beta_min_max


_PLACEHOLDER_NAME = 'space_logits'

_EXTRA_DIMS = [None, None]

_placeholder_factory_map = {
  discrete.Discrete: lambda space: array_ops.placeholder(
      space.dtype, _EXTRA_DIMS + [space.n], name=_PLACEHOLDER_NAME),
  multi_discrete.MultiDiscrete: lambda space: array_ops.placeholder(
      space.dtype, _EXTRA_DIMS + list(space.shape), name=_PLACEHOLDER_NAME),
  multi_binary.MultiBinary: lambda space: array_ops.placeholder(
      space.dtype, _EXTRA_DIMS + list(space.shape), name=_PLACEHOLDER_NAME),
  box.Box: lambda space: array_ops.placeholder(
      space.dtype, _EXTRA_DIMS + list(space.shape), name=_PLACEHOLDER_NAME),
}


class DistributionWithLogits(
    collections.namedtuple('DistributionWithLogits',
                           ['distribution', 'logits'])):
  pass


def placeholder_from_gym_space(space, name='SpacePlaceholder'):
  """Determines a placeholder from the `gym.Space`.

  Arguments:
    space: a `gym.Space` instance (i.e. `env.action_space`)
    name: Python `str` name prefixed to Ops created.

  Raises:
    `TypeError` when space is not a `gym.Space` instance.

  Returns:
    Either one of the following: `tf.Tensor`, `tuple` or `dict` of `tf.Tensor`
  """
  with ops.name_scope(name):
    if isinstance(space, discrete.Discrete):
      return _placeholder_factory_map[discrete.Discrete](space)
    elif isinstance(space, multi_discrete.MultiDiscrete):
      return _placeholder_factory_map[multi_discrete.MultiDiscrete](space)
    elif isinstance(space, multi_binary.MultiBinary):
      return _placeholder_factory_map[multi_binary.MultiBinary](space)
    elif isinstance(space, box.Box):
      return _placeholder_factory_map[box.Box](space)
    elif isinstance(space, tuple_space.Tuple):
      return tuple(placeholder_from_gym_space(val, name='tuple_{}'.format(idx))
                   for idx, val in enumerate(space.spaces))
    elif isinstance(space, dict_space.Dict):
      return {key: placeholder_from_gym_space(val, name='{}'.format(key))
              for key, val in space.spaces.items()}
  raise TypeError('`space` not supported: {}'.format(type(space)))


# TODO(wenkesj): use currying here and accept logits as an argument to create
#                arbitrary distributions. This will allow distributions to be parameterized
#                way easier and seperates the logic so it's easier to read and comprehend
#                what is going on.
def distribution_from_gym_space(space, logits=None, name='SpaceDistribution', trainable=True):
  """Determines a parameterized `tf.distribution.Distribution` from the `gym.Space`.

  Arguments:
    space: a `gym.Space` instance (i.e. `env.action_space`)
    logits: optional `list` of `tf.Tensor` to be used instead of creating them.
    name: Python `str` name prefixed to Ops created.

  Raises:
    `TypeError` when space is not a `gym.Space` instance.

  Returns:
    Either one of the following: , `tuple` or `dict` of `DistributionWithLogits`, or
        just `DistributionWithLogits`.
  """
  assert_utils.assert_true(
      isinstance(space, Space),
      '`space` must be an instance of `gym.Space`')

  with ops.name_scope(name):
    if isinstance(space, discrete.Discrete):
      if logits and isinstance(logits[0], ops.Tensor):
        logits = _dense_projection(logits[0], [space.n], trainable=trainable)
      else:
        logits = _placeholder_factory_map[discrete.Discrete](space)
      distribution = categorical.Categorical(logits=math_ops.cast(logits, dtypes.float32))
      return DistributionWithLogits(
          distribution=distribution,
          logits=logits)

    elif isinstance(space, multi_discrete.MultiDiscrete):
      if logits and isinstance(logits[0], ops.Tensor):
        logits = _dense_projection(logits[0], space.shape, trainable=trainable)
      else:
        logits = _placeholder_factory_map[multi_discrete.MultiDiscrete](space)
      distribution = categorical.Categorical(logits=math_ops.cast(logits, dtypes.float32))
      return DistributionWithLogits(
          distribution=distribution,
          logits=logits)

    elif isinstance(space, multi_binary.MultiBinary):
      if logits and isinstance(logits[0], ops.Tensor):
        logits = _dense_projection(logits[0], space.shape, trainable=trainable)
      else:
        logits = _placeholder_factory_map[multi_binary.MultiBinary](space)
      distribution = bernoulli.Bernoulli(logits=logits)
      return DistributionWithLogits(
          distribution=distribution,
          logits=logits)

    elif isinstance(space, box.Box):
      if logits and isinstance(logits[0], ops.Tensor):
        logits = _dense_projection(logits[0], space.shape, trainable=trainable)
      else:
        logits = _placeholder_factory_map[box.Box](space)

      flat_shape = array_utils.product(space.shape)
      shape = array_ops.shape(logits)
      logits = gen_array_ops.reshape(
          logits, [shape[0], shape[1], flat_shape])

      log_eps = math.log(distribution_utils.epsilon)

      alpha = core.dense(logits, flat_shape, use_bias=False, trainable=trainable)
      alpha = clip_ops.clip_by_value(alpha, log_eps, -log_eps)
      alpha = math_ops.log(math_ops.exp(alpha) + 1.0) + 1.0
      alpha = gen_array_ops.reshape(alpha, shape)

      beta = core.dense(logits, flat_shape, use_bias=False, trainable=trainable)
      beta = clip_ops.clip_by_value(beta, log_eps, -log_eps)
      beta = math_ops.log(math_ops.exp(beta) + 1.0) + 1.0
      beta = gen_array_ops.reshape(beta, shape)

      distribution = beta_min_max.BetaMinMax(
          concentration1=alpha,
          concentration0=beta,
          min_value=space.low,
          max_value=space.high)
      return DistributionWithLogits(
          distribution=distribution,
          logits=logits)

    elif isinstance(space, tuple_space.Tuple):
      if not logits:
        logits = [None] * len(space.spaces)
      return tuple(distribution_from_gym_space(
          val, logits=[logit], name='tuple_{}'.format(idx))
          for idx, (val, logit) in enumerate(zip(space.spaces, logits)))

    elif isinstance(space, dict_space.Dict):
      if not logits:
        logits = [None] * len(space.spaces)
      return {key: distribution_from_gym_space(
          val, logits=[logit], name='{}'.format(key))
          for (key, val), logit in zip(space.spaces.items(), logits)}

  raise TypeError('`space` not supported: {}'.format(type(space)))


def _dense_projection(inputs, shape, trainable=True):
  flat_shape = array_utils.product(shape)
  target_shape = _EXTRA_DIMS + shape

  input_shape = inputs.get_shape().as_list()

  input_dims = len(input_shape)
  expected_dims = len(target_shape)
  assert_utils.assert_true(
      input_dims == expected_dims,
      ', '.join([
          '`inputs` must have the same number of dims as the number expected dims.'
          'expected = {}, actual = {}'.format(input_dims, expected_dims)]))

  if not array_utils.all_equal(input_shape, target_shape):
    input_shape_ = array_ops.shape(inputs)
    if len(input_shape) > 3:
      inputs = gen_array_ops.reshape(
          inputs, [input_shape_[0], input_shape_[1], -1])
    inputs = core.dense(inputs, flat_shape, use_bias=False, trainable=trainable)
    inputs = gen_array_ops.reshape(
        inputs, [input_shape_[0], input_shape_[1]] + shape)
  return inputs
