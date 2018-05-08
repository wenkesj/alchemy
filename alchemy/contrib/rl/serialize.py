# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import dtypes
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


def serialize_replay_feature(values, dtype):
  """Converts `values` to a list of `tf.train.Feature`."""
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


def serialize_replay(replay,
                     state_dtype,
                     action_dtype,
                     action_value_dtype,
                     reward_dtype):
  """Returns a `tf.train.SequenceExample` for the given `ay.contrib.rl.Replay` instance.

  Arguments:
    replay: `ay.contrib.rl.Replay` instance.
    state_dtype: dtype of the state space.
    action_dtype: dtype of the action space.
    action_value_dtype: dtype of the action-values space.
    reward_dtype: dtype of the reward space.

  Returns:
    A `tf.train.SequenceExample` containing info from the `ay.contrib.rl.Replay`.

  Raises:
    `AssertionError` when replay is not an `ay.rl.Replay` instance.
  """
  assert_utils.assert_true(
      isinstance(replay, experience.Replay),
      '`replay` must be an instance of `ay.contrib.rl.Replay`')

  feature_list = {
    'state': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.state, dtype=state_dtype)),
    'next_state': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.next_state, dtype=state_dtype)),
    'action': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.action, dtype=action_dtype)),
    'action_value': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.action_value, dtype=action_value_dtype)),
    'reward': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.reward, dtype=reward_dtype)),
    'terminal': tf.train.FeatureList(
        feature=serialize_replay_feature(
            replay.terminal, dtype=dtypes.bool)),
    'sequence_length': tf.train.FeatureList(
        feature=serialize_replay_feature(
            [replay.sequence_length], dtype=dtypes.int32)),
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(feature_lists=feature_lists)
