# -*- coding: utf-8 -*-
from __future__ import absolute_import

from gym.core import Env
from gym.core import Space

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import errors_impl

from alchemy.utils import assert_utils
from alchemy.utils import array_utils
from alchemy.utils import distribution_utils
from alchemy.utils import type_utils
from alchemy.contrib.rl import serialize


class ReplayStream(object):

  """
  A duplex stream that serializing information to/from TensorFlow to/from an environment.
  """

  def __init__(self,
               state_shape, state_dtype,
               action_shape, action_dtype,
               action_value_shape, action_value_dtype,
               reward_shape, reward_dtype,
               with_values=False):
    """Creates a new instance by the given shapes and dtypes.

    Arguments:
      state_shape: shape of the state space.
      state_dtype: dtype of the state space.
      action_shape: shape of the action space.
      action_dtype: dtype of the action space.
      action_value_shape: shape of the action-values space.
      action_value_dtype: dtype of the action-values space.
      reward_shape: shape of the reward space.
      reward_dtype: dtype of the reward space.
      with_values: Python `bool` for recording values.
    """
    self.state_shape = list(state_shape)
    self.state_dtype = type_utils.safe_tf_dtype(state_dtype)
    self.action_shape = list(action_shape)
    self.action_dtype = type_utils.safe_tf_dtype(action_dtype)
    self.action_value_shape = list(action_value_shape)
    self.action_value_dtype = type_utils.safe_tf_dtype(action_value_dtype)
    self.reward_shape = list(reward_shape)
    self.reward_dtype = type_utils.safe_tf_dtype(reward_dtype)
    self.with_values = with_values

  @classmethod
  def from_distributions(cls,
                         state_distribution, action_distribution,
                         reward_shape=[], reward_dtype=dtypes.float32,
                         with_values=False):
    """Construct a `alchemy.contrib.rl.ReplayStream` from a `gym.Env`.

    Arguments:
      env: a `gym.Env` instance that has `action_space` and `observation_space` properties.
      state_distribution: distribution of the state space.
      action_distribution: distribution of the action space.
      reward_shape: shape representing the reward for a chosen action.
      reward_dtype: dtype representing the reward for a chosen action.
      with_values: Python `bool` for recording values.

    Returns:
      A `ay.contrib.rl.ReplayStream`.
    """
    assert_utils.assert_true(
        distribution_utils.is_distribution(state_distribution),
        '`state_distribution` must be an instance of `tf.distributions.Distribution`')
    assert_utils.assert_true(
        distribution_utils.is_distribution(action_distribution),
        '`action_distribution` must be an instance of `tf.distributions.Distribution`')

    state_shape, state_dtype = distribution_utils.logits_shape_and_dtype(
        state_distribution)
    action_value_shape, action_value_dtype = distribution_utils.logits_shape_and_dtype(
        action_distribution)
    action_shape, action_dtype = distribution_utils.sample_shape_and_dtype(
        action_distribution)
    if isinstance(reward_shape, tensor_shape.TensorShape):
      reward_shape = reward_shape.as_list()

    return cls(
        state_shape, state_dtype,
        action_shape, action_dtype,
        action_value_shape, action_value_dtype,
        reward_shape, type_utils.safe_tf_dtype(reward_dtype),
        with_values=with_values)

  def serialize_replay(self, replay):
    """Returns a `tf.train.SequenceExample` for the given replay instance.

    Arguments:
      replay: `ay.contrib.rl.Replay` or `ay.contrib.rl.ReplayWithValues` instance.

    Returns:
      A `tf.train.SequenceExample` from `ay.contrib.rl.Replay` or `ay.contrib.rl.ReplayWithValues`

    Raises:
      `AssertionError` when replay is not an `ay.contrib.rl.Replay` or
          `ay.contrib.rl.ReplayWithValues` instance.
    """
    return serialize.serialize_replay(
        replay,
        self.state_dtype,
        self.action_dtype,
        self.action_value_dtype,
        self.reward_dtype,
        with_values=self.with_values)

  def read(self, limit=1):
    """Read a `tf.train.SequenceExample` from memory.

    Arguments:
      limit: maximum sequence length of all sequences.

    Returns:
      A `tf.train.SequenceExample` or `None`.

    Raises:
      An `tf.errors.OutOfRangeError`.
    """
    raise NotImplementedError('Must implement `read` method.')

  def write(self, replay):
    """Write an `ay.contrib.rl.Replay` or `ay.contrib.rl.ReplayWithValues` instance to memory.

    Arguments:
      replay: An `ay.contrib.rl.Replay` or `ay.contrib.rl.ReplayWithValues` instance.
    """
    raise NotImplementedError('Must implement `write` method.')


class Stack(ReplayStream):

  """
  A Stack replay memory.
  """

  def __init__(self, *args, **kwargs):
    super(Stack, self).__init__(*args, **kwargs)
    self.memory = []

  def write(self, replay):
    self.memory.append(replay)

  def read(self, limit=1):
    if not self.memory:
      raise errors_impl.OutOfRangeError()
    memory = self.memory.pop()
    if len(memory[0]) > limit:
      partitions = memory.partition(limit, allow_overflow=True)
      memory = partitions.pop()
      self.memory.extend(partitions)
    return self.serialize_replay(memory)


class Queue(ReplayStream):

  """
  A Queue replay memory.
  """

  def __init__(self, *args, **kwargs):
    super(Queue, self).__init__(*args, **kwargs)
    self.memory = []

  def write(self, replay):
    self.memory.append(replay)

  def read(self, limit=1):
    if not self.memory:
      raise errors_impl.OutOfRangeError()
    memory = self.memory.pop(0)
    if len(memory[0]) > limit:
      partitions = memory.partition(limit, allow_overflow=True)
      memory = partitions.pop(0)
      self.memory = partitions + self.memory
    return self.serialize_replay(memory)


class Uniform(ReplayStream):

  """
  A uniform-sampling replay memory.
  """

  def __init__(self, *args, **kwargs):
    super(Uniform, self).__init__(*args, **kwargs)
    self.memory = []

  def write(self, replay):
    self.memory.append(replay)

  def read(self, limit=1):
    if not self.memory:
      raise errors_impl.OutOfRangeError()
    index = np.random.randint(len(self.memory))
    memory = self.memory.pop(index)

    if len(memory[0]) >= limit:
      partitions = memory.partition(limit, allow_overflow=True)
      index = np.random.randint(len(partitions))
      memory = partitions.pop(index)
      self.memory = partitions + self.memory

    return self.serialize_replay(memory)
