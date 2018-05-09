# -*- coding: utf-8 -*-
from __future__ import absolute_import

from gym.core import Env
from gym.core import Space

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl

from alchemy.utils import assert_utils
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
               reward_shape, reward_dtype):
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
    """
    self.state_shape = list(state_shape)
    self.state_dtype = type_utils.safe_tf_dtype(state_dtype)
    self.action_shape = list(action_shape)
    self.action_dtype = type_utils.safe_tf_dtype(action_dtype)
    self.action_value_shape = list(action_value_shape)
    self.action_value_dtype = type_utils.safe_tf_dtype(action_value_dtype)
    self.reward_shape = list(reward_shape)
    self.reward_dtype = type_utils.safe_tf_dtype(reward_dtype)

  @classmethod
  def from_gym_env(cls, env,
                   action_value_shape, action_value_dtype,
                   reward_shape=[], reward_dtype=dtypes.float32):
    """Construct a `alchemy.contrib.rl.ReplayStream` from a `gym.Env`.

    Arguments:
      env: a `gym.Env` instance that has `action_space` and `observation_space` properties.
      action_value_shape: shape representing the logits for a chosen action.
      action_value_dtype: dtype representing the logits for a chosen action.
      reward_shape: shape representing the reward for a chosen action.
      reward_dtype: dtype representing the reward for a chosen action.

    Returns:
      A `ay.contrib.rl.ReplayStream`.
    """
    assert_utils.assert_true(
        isinstance(env, Env),
        '`env` must be an instance of `gym.Env`')
    assert_utils.assert_true(
        env.action_space is not None,
        '`env.action_space` property must be set.')
    assert_utils.assert_true(
        isinstance(env.action_space, Space),
        '`env.action_space` must be an instance of `gym.Space`')
    assert_utils.assert_true(
        env.observation_space is not None,
        '`env.observation_space` property must be set.')
    assert_utils.assert_true(
        isinstance(env.observation_space, Space),
        '`env.observation_space` must be an instance of `gym.Space`')

    state_shape = env.observation_space.shape
    state_dtype = type_utils.safe_tf_dtype(env.observation_space.dtype)
    action_shape = env.action_space.shape
    action_dtype = type_utils.safe_tf_dtype(env.action_space.dtype)

    return cls(
        state_shape, state_dtype,
        action_shape, action_dtype,
        action_value_shape, type_utils.safe_tf_dtype(action_value_dtype),
        reward_shape, type_utils.safe_tf_dtype(reward_dtype))

  def serialize_replay(self, replay):
    """Returns a `tf.train.SequenceExample` for the given `ay.contrib.rl.Replay` instance.

    Arguments:
      replay: `ay.contrib.rl.Replay` instance.

    Returns:
      A `tf.train.SequenceExample` containing info from the `ay.contrib.rl.Replay`.

    Raises:
      `AssertionError` when replay is not an `ay.rl.Replay` instance.
    """
    return serialize.serialize_replay(
        replay,
        self.state_dtype,
        self.action_dtype,
        self.action_value_dtype,
        self.reward_dtype)

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
    """Write and serialize an `ay.contrib.rl.Replay` instance to memory.

    Arguments:
      replay: An `ay.contrib.rl.Replay` instance.
    """
    raise NotImplementedError('Must implement `write` method.')


class SimpleReplayStream(ReplayStream):

  """
  A simple, FIFO replay memory.
  """

  def __init__(self, *args, **kwargs):
    super(SimpleReplayStream, self).__init__(*args, **kwargs)
    self.memory = []

  def write(self, replay):
    self.memory.append(self.serialize_replay(replay))

  def read(self, limit=1):
    if not self.memory:
      raise errors_impl.OutOfRangeError()
    return self.memory.pop(0)
