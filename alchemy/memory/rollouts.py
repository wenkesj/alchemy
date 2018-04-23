# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from time import sleep

from alchemy.memory import trajectory
from alchemy.multiprocessing import pool
from alchemy.utils import array_utils
from alchemy.utils import type_utils


# TODO(wenkesj): add BETTER docstring
def rollout(env, step_fn,
            initial_internals=None,
            max_rollout_steps=-1,
            max_trajectory_steps=-1):
  """Record histroy of an env by a step_fn, returns a list of Trajectory(s)."""
  traj = trajectory.Trajectory([])
  next_state = env.reset()
  stoppable = max_rollout_steps > 0

  internals = initial_internals
  step = 0
  while True:
    if stoppable:
      if step >= max_rollout_steps:
        break
    state = next_state
    action, values, next_internals = step_fn(state, internals)
    next_state, reward, terminal, info = env.step(action)

    traj.add(
        trajectory.Transition(state=state,
                   action=action,
                   values=values,
                   reward=reward,
                   terminal=terminal,
                   info=info))
    internals = next_internals
    step += 1

    if terminal:
      break

  if max_trajectory_steps > 0:
    if step > max_trajectory_steps:
      trajs = []
      for part in array_utils.partition(traj.transitions, max_trajectory_steps):
        traj_part = trajectory.Trajectory([])
        for pack in zip(*part):
          traj_part.add(trajectory.Transition(*pack))
        trajs.append(traj_part)
      return trajs
  return [traj]


# TODO(wenkesj): add docstring
def rollout_to_src(src, env, step_fn,
                   initial_internals=None,
                   num_episodes=1,
                   max_rollout_steps=-1,
                   max_trajectory_steps=-1):
  episode = 0
  while True:
    if episode >= num_episodes:
      break
    trajectories = rollout(
        env, step_fn,
        initial_internals=initial_internals,
        max_rollout_steps=max_rollout_steps,
        max_trajectory_steps=max_trajectory_steps)

    for traj in trajectories:
      src.write(traj)
    episode += 1
  env.close()


# TODO(wenkesj): add docstring
class RolloutPool(pool.ThreadPool):

  def __init__(self, create_env_fn, num_envs=1, num_threads=1):
    super(RolloutPool, self).__init__(num_threads)
    self._create_env_fn = create_env_fn
    self._num_envs = num_envs

  def __call__(self, src, step_fn,
               synchronous=False,
               initial_internals=None,
               num_episodes=1,
               max_rollout_steps=-1,
               max_trajectory_steps=-1):
    assert num_episodes >= 0

    envs = []
    for idx in range(self._num_envs):
      envs.append(self._create_env_fn())

    for env in envs:
      self.add_task(
          rollout_to_src,
          src=src, env=env,
          step_fn=step_fn,
          initial_internals=initial_internals,
          num_episodes=num_episodes,
          max_rollout_steps=max_rollout_steps,
          max_trajectory_steps=max_trajectory_steps)

    if synchronous:
      self.wait_completion()


# TODO(wenkesj): add BETTER docstring
def rollout_dataset(src,
                    batch_size=1,
                    max_sequence_length=200,
                    min_sequence_length=2,
                    name='replay_dataset'):
  """Create a dataset from a replay memory source and stream it through a tf Dataset."""
  with tf.name_scope(name):
    state_dtype = src.state_dtype
    state_shape = src.state_shape
    action_dtype = src.action_dtype
    action_shape = src.action_shape
    action_value_dtype = src.action_value_dtype
    action_value_shape = src.action_value_shape

    def src_stream():
      while True:
        sleep(.05)
        if len(src) < 1:
          continue

        traj = src.read()
        if traj.size < min_sequence_length:
          continue

        state, action, value, reward, terminal, _ = zip(*traj.transitions)
        sequence_length = np.asarray([min(max_sequence_length, traj.size)])
        state = np.asarray(
            array_utils.list_pad_or_truncate(
                list(state), max_sequence_length, np.zeros(state_shape)))
        action = np.asarray(
            array_utils.list_pad_or_truncate(
                list(action), max_sequence_length, np.zeros(action_shape)))
        value = np.asarray(
            array_utils.list_pad_or_truncate(
                list(value), max_sequence_length, np.zeros(action_value_shape)))
        reward = np.asarray(
            array_utils.list_pad_or_truncate(
                list(reward), max_sequence_length, 0.))
        terminal = np.asarray(
            array_utils.list_pad_or_truncate(
                list(terminal), max_sequence_length, True))
        yield (state, action, value, reward, terminal, sequence_length)

    dtypes_and_shapes = [
        (type_utils.safe_tf_dtype(state_dtype), [None] + state_shape), # state
        (type_utils.safe_tf_dtype(action_dtype), [None] + action_shape), # action
        (type_utils.safe_tf_dtype(action_value_dtype), [None] + action_value_shape), # value
        (tf.float32, [None]), # reward
        (tf.bool, [None]), # terminal
        (tf.int32, [1]), # sequence_length
    ]

    ds = tf.data.Dataset.from_generator(src_stream, *zip(*dtypes_and_shapes))
    ds = ds.batch(batch_size)
    iterator = ds.make_one_shot_iterator()

    state, action, value, reward, terminal, sequence_length = iterator.get_next()
    sequence_length = tf.squeeze(sequence_length, axis=-1) - 1
    next_state = state[:, 1:, ...]
    state = state[:, :-1, ...]
    action = action[:, 1:, ...]
    value = value[:, 1:, ...]
    reward = reward[:, 1:]
    terminal = terminal[:, 1:]

    return state, next_state, action, value, reward, terminal, sequence_length
