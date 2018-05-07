# -*- coding: utf-8 -*-
from __future__ import absolute_import

from gym.core import Env
from gym.core import Space

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.util import compat
import tensorflow as tf

from alchemy.utils import assert_utils
from alchemy.utils import type_utils
from alchemy.contrib.rl import experience


type_to_feature = {
  dtypes.float16: ('float_list', tf.train.FloatList, np.float32),
  dtypes.float32: ('float_list', tf.train.FloatList, np.float32),
  dtypes.float64: ('float_list', tf.train.FloatList, np.float32),
  dtypes.bfloat16: ('float_list', tf.train.FloatList, np.float32),
  dtypes.int8: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.uint8: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.uint16: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.uint32: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.uint64: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.int16: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.int32: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.int64: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.bool: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.string: ('bytes_list', tf.train.BytesList, np.dtype(bytes)),
  dtypes.qint8: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.quint8: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.qint16: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.quint16: ('int64_list', tf.train.Int64List, np.int64),
  dtypes.qint32: ('int64_list', tf.train.Int64List, np.int64),
}


class ReplayStream(object):

  """
  A way to handle serializing information to TensorFlow from an environment.
  """

  def __init__(self,
               state_shape, state_dtype,
               action_shape, action_dtype,
               action_value_shape, action_value_dtype,
               reward_shape, reward_dtype):
    self.state_shape = state_shape
    self.state_dtype = type_utils.safe_tf_dtype(state_dtype)
    self.action_shape = action_shape
    self.action_dtype = type_utils.safe_tf_dtype(action_dtype)
    self.action_value_shape = action_value_shape
    self.action_value_dtype = type_utils.safe_tf_dtype(action_value_dtype)
    self.reward_shape = reward_shape
    self.reward_dtype = type_utils.safe_tf_dtype(reward_dtype)

  @classmethod
  def from_gym_env(cls, env,
                   action_value_shape, action_value_dtype,
                   reward_shape=[], reward_dtype=dtypes.float32):
    """Construct a `alchemy.contrib.rl.ReplayStream` from a `gym.Env`.

    Args:
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

  def serialize_replay_feature(self, values, dtype):
    feature_list_name, feature_list_fn, feature_dtype = type_to_feature[
        type_utils.safe_tf_dtype(dtype)]
    value_features = []
    for value in values:
      is_bytes_or_number = isinstance(value, bytes) or assert_utils.is_number(value)
      if not assert_utils.is_iterable(value) or is_bytes_or_number:
        value = [value]
      if isinstance(value, np.ndarray):
        new_value = value.astype(feature_dtype)
      else:
        new_value = np.asarray(value, dtype=feature_dtype)
      value_feature = tf.train.Feature(**{
          feature_list_name: feature_list_fn(value=new_value.ravel())})
      value_features.append(value_feature)
    return value_features

  def serialize_replay(self, replay):
    """Returns a `tf.train.SequenceExample` for the given `ay.contrib.rl.Replay` instance.

    Args:
      replay: `ay.contrib.rl.Replay` instance.

    Returns:
      A `tf.train.SequenceExample` containing info from the `ay.contrib.rl.Replay`.
    """
    assert_utils.assert_true(
        isinstance(replay, experience.Replay),
        '`replay` must be an instance of `ay.contrib.rl.Replay`')

    feature_list = {
      'state': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.state, dtype=self.state_dtype)),
      'next_state': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.next_state, dtype=self.state_dtype)),
      'action': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.action, dtype=self.action_dtype)),
      'action_value': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.action_value, dtype=self.action_value_dtype)),
      'reward': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.reward, dtype=self.reward_dtype)),
      'terminal': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              replay.terminal, dtype=dtypes.bool)),
      'sequence_length': tf.train.FeatureList(
          feature=self.serialize_replay_feature(
              [replay.sequence_length], dtype=dtypes.int32)),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

  def read(self, limit=1):
    raise NotImplementedError('Must implement `read` method.')

  def write(self, replay):
    raise NotImplementedError('Must implement `write` method.')


class SimpleReplayStream(ReplayStream):

  """
  Replay 101.
  """

  def __init__(self, *args, **kwargs):
    super(SimpleReplayStream, self).__init__(*args, **kwargs)
    self.memory = []

  def write(self, replay):
    self.memory.append(self.serialize_replay(replay))

  def read(self, limit=1):
    if not self.memory:
      return None
    return self.memory.pop(0)
