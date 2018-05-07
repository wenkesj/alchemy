# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import errors_impl
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from alchemy.utils import array_utils
from alchemy.utils import assert_utils
from alchemy.utils import sequence_utils
from alchemy.utils import type_utils
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams


def ReplayDataset(replay_stream, max_sequence_length=200, name=None):
  """Creates a `tf.data.Dataset` from a `ay.contrib.rl.ReplayStream` instance."""
  assert_utils.assert_true(
      isinstance(replay_stream, streams.ReplayStream),
      '`replay_stream` must be an instance of `ay.contrib.rl.ReplayStream`')

  with ops.name_scope(name or 'replay_dataset'):
    state_shape = list(replay_stream.state_shape)
    state_dtype = replay_stream.state_dtype
    action_shape = list(replay_stream.action_shape)
    action_dtype = replay_stream.action_dtype
    action_value_shape = list(replay_stream.action_value_shape)
    action_value_dtype = replay_stream.action_value_dtype
    reward_shape = list(replay_stream.reward_shape)
    reward_dtype = replay_stream.reward_dtype

    replay_dtypes = {
      'state': state_dtype,
      'next_state': state_dtype,
      'action': action_dtype,
      'action_value': action_value_dtype,
      'reward': reward_dtype,
      'terminal': dtypes.bool,
      'sequence_length': dtypes.int32,
    }

    def convert_to_safe_feature_type(dtype):
      return type_utils.safe_tf_dtype(streams.type_to_feature[dtype][-1])

    replay_features = {
      'state': parsing_ops.FixedLenSequenceFeature(
          shape=state_shape, dtype=convert_to_safe_feature_type(state_dtype)),
      'next_state': parsing_ops.FixedLenSequenceFeature(
          shape=state_shape, dtype=convert_to_safe_feature_type(state_dtype)),
      'action': parsing_ops.FixedLenSequenceFeature(
          shape=action_shape, dtype=convert_to_safe_feature_type(action_dtype)),
      'action_value': parsing_ops.FixedLenSequenceFeature(
          shape=action_value_shape, dtype=convert_to_safe_feature_type(action_value_dtype)),
      'reward': parsing_ops.FixedLenSequenceFeature(
          shape=reward_shape, dtype=convert_to_safe_feature_type(reward_dtype)),
      'terminal': parsing_ops.FixedLenSequenceFeature(
          shape=[], dtype=convert_to_safe_feature_type(dtypes.bool)),
      'sequence_length': parsing_ops.FixedLenSequenceFeature(
          shape=[], dtype=convert_to_safe_feature_type(dtypes.int32)),
    }

    def convert_and_fix_dtypes(replay):
      """Cast dtypes back to their original types."""
      fixed_replay = {}
      for k, v in replay.items():
        fixed_replay[k] = math_ops.cast(v, dtype=replay_dtypes[k])
      return fixed_replay

    def generator():
      """Create `tf.Tensor`s from the `ay.contrib.rl.ReplayStream` instance."""
      while True:
        replay_example = None
        try:
          replay_example = replay_stream.read(limit=max_sequence_length)
        except:
          raise errors_impl.OutOfRangeError()
        if replay_example is None:
          raise errors_impl.OutOfRangeError()
        yield replay_example.SerializeToString()

    def serialize_map(replay_example_str):
      """Parse each example string to `tf.Tensor`."""
      _, replay = parsing_ops.parse_single_sequence_example(
          replay_example_str, sequence_features=replay_features)
      return convert_and_fix_dtypes(replay)

    def pad_or_truncate_map(replay):
      """Truncate or pad replays."""
      replay = experience.Replay(**replay)
      sequence_length = math_ops.minimum(
          max_sequence_length, replay.sequence_length)
      sequence_length.set_shape([1])

      state = sequence_utils.pad_or_truncate(
          replay.state, max_sequence_length,
          axis=0, pad_value=0)
      state.set_shape([max_sequence_length] + state_shape)

      next_state = sequence_utils.pad_or_truncate(
          replay.next_state, max_sequence_length,
          axis=0, pad_value=0)
      next_state.set_shape([max_sequence_length] + state_shape)

      action = sequence_utils.pad_or_truncate(
          replay.action, max_sequence_length,
          axis=0, pad_value=0)
      action.set_shape([max_sequence_length] + action_shape)

      action_value = sequence_utils.pad_or_truncate(
          replay.action_value, max_sequence_length,
          axis=0, pad_value=0)
      action_value.set_shape([max_sequence_length] + action_value_shape)

      reward = sequence_utils.pad_or_truncate(
          replay.reward, max_sequence_length,
          axis=0, pad_value=0)
      reward.set_shape([max_sequence_length] + reward_shape)

      terminal = sequence_utils.pad_or_truncate(
          replay.terminal, max_sequence_length,
          axis=0, pad_value=ops.convert_to_tensor(False))
      terminal.set_shape([max_sequence_length])

      return experience.Replay(
          state=state,
          next_state=next_state,
          action=action,
          action_value=action_value,
          reward=reward,
          terminal=terminal,
          sequence_length=sequence_length)

    dataset = dataset_ops.Dataset.from_generator(generator, dtypes.string)
    dataset = dataset.map(serialize_map)
    return dataset.map(pad_or_truncate_map)
