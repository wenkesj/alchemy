# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import errors_impl
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from alchemy.utils import array_utils
from alchemy.utils import assert_utils
from alchemy.utils import sequence_utils
from alchemy.utils import type_utils
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl import serialize


def ReplayDataset(replay_stream, max_sequence_length=200, name=None):
  """Creates a `tf.data.Dataset` from a `ay.contrib.rl.ReplayStream` instance.

  Arguments:
    replay_stream: `ay.contrib.rl.ReplayStream` instance. Must implement `replay_stream.read`.
        The method is called `replay_stream.read(limit=max_sequence_length)` each time an instance
        is requested by the dataset. This method should return `None` or raise an
        `tf.errors.OutOfRangeError` when the stream is done and execution of the dataset should stop.
        `replay_stream.read` should always return a `tf.SequenceExample` proto.

  Returns:
    A `tf.data.Dataset`.

  Raises:
    An `tf.errors.OutOfRangeError` when the stream returns a `None` or raises
        `tf.errors.OutOfRangeError`.
  """
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

    if replay_stream.with_values:
      replay_dtypes['value'] = reward_dtype

    def convert_to_safe_feature_type(dtype):
      return type_utils.safe_tf_dtype(serialize.type_to_feature[dtype][-1])

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

    if replay_stream.with_values:
      replay_features['value'] = parsing_ops.FixedLenSequenceFeature(
          shape=reward_shape, dtype=convert_to_safe_feature_type(reward_dtype))

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
          yield ""
        else:
          yield replay_example.SerializeToString()

    def serialize_map(replay_example_str):
      """Parse each example string to `tf.Tensor`."""
      try:
        assert_op = control_flow_ops.Assert(
            replay_example_str != "",
            [replay_example_str])
        with ops.control_dependencies([assert_op]):
          _, replay = parsing_ops.parse_single_sequence_example(
              replay_example_str, sequence_features=replay_features)
      except errors_impl.InvalidArgumentError:
        raise errors_impl.OutOfRangeError()

      return convert_and_fix_dtypes(replay)

    def pad_or_truncate_map(replay):
      """Truncate or pad replays."""
      with_values = 'value' in replay

      if with_values:
        replay = experience.ReplayWithValues(**replay)
      else:
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

      if with_values:
        value = sequence_utils.pad_or_truncate(
            replay.value, max_sequence_length,
            axis=0, pad_value=0)
        value.set_shape([max_sequence_length] + reward_shape)

        return experience.ReplayWithValues(
            state=state,
            next_state=next_state,
            action=action,
            action_value=action_value,
            value=value,
            reward=reward,
            terminal=terminal,
            sequence_length=sequence_length)

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
