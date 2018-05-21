# -*- coding: utf-8 -*-
from __future__ import absolute_import

from alchemy.contrib.rl import experience


# TODO(wenkesj): cutdown on repeated code, make this more modular.

def rollout_on_gym_env(sess,
                       env,
                       state_ph,
                       deterministic_ph,
                       action_value_op,
                       action_op,
                       num_episodes=1,
                       deterministic=False,
                       stream=None,
                       save_replay=True):
  if stream is None and save_replay:
    raise ValueError('missing `stream` to `save_replay`.')

  rewards = 0.
  for episode in range(num_episodes):
    experiences = []
    next_state = env.reset()
    while True:
      state = next_state
      action_value, action = sess.run(
          (action_value_op, action_op),
          feed_dict={
              state_ph: [[state]],
              deterministic_ph: deterministic
          })
      next_state, reward, terminal, _ = env.step(action)
      experiences.append(experience.Experience(
          state, next_state,
          action, action_value,
          reward, terminal))
      if terminal:
        break
    replay = experience.Replay(
        *zip(*experiences),
        sequence_length=len(experiences))
    if save_replay:
      stream.write(replay)
    rewards += sum(replay.reward)
  return rewards


def rollout_with_values_on_gym_env(sess,
                                   env,
                                   state_ph,
                                   deterministic_ph,
                                   action_value_op,
                                   action_op,
                                   value_op,
                                   num_episodes=1,
                                   deterministic=False,
                                   stream=None,
                                   save_replay=True):
  if stream is None and save_replay:
    raise ValueError('missing `stream` to `save_replay`.')

  rewards = 0.
  for episode in range(num_episodes):
    experiences = []
    next_state = env.reset()
    while True:
      state = next_state
      action_value, action, value = sess.run(
          (action_value_op, action_op, value_op),
          feed_dict={
              state_ph: [[state]],
              deterministic_ph: deterministic
          })

      next_state, reward, terminal, _ = env.step(action)
      experiences.append(experience.ExperienceWithValues(
          state, next_state,
          action, action_value,
          value,
          reward, terminal))

      if terminal:
        break
    replay = experience.ReplayWithValues(
        *zip(*experiences),
        sequence_length=len(experiences))
    if save_replay:
      stream.write(replay)
    rewards += sum(replay.reward)
  return rewards


def rollout_stateful_with_values_on_gym_env(sess,
                                            env,
                                            state_ph,
                                            deterministic_ph,
                                            action_value_op,
                                            action_op,
                                            value_op,
                                            internal_state_ph,
                                            internal_state_op,
                                            zero_state,
                                            num_episodes=1,
                                            deterministic=False,
                                            stream=None,
                                            save_replay=True):
  if stream is None and save_replay:
    raise ValueError('missing `stream` to `save_replay`.')

  rewards = 0.
  for episode in range(num_episodes):
    internal_state = sess.run(zero_state(1, dtype=state_ph.dtype))
    experiences = []
    next_state = env.reset()
    while True:
      state = next_state
      action_value, action, value, internal_state = sess.run(
          (action_value_op, action_op, value_op, internal_state_op),
          feed_dict={
              state_ph: [[state]],
              **{k: v for k, v in zip(internal_state_ph, internal_state)},
              deterministic_ph: deterministic
          })

      next_state, reward, terminal, _ = env.step(action)
      experiences.append(experience.ExperienceWithValues(
          state, next_state,
          action, action_value,
          value,
          reward, terminal))

      if terminal:
        break
    replay = experience.ReplayWithValues(
        *zip(*experiences),
        sequence_length=len(experiences))
    if save_replay:
      stream.write(replay)
    rewards += sum(replay.reward)
  return rewards


def rollout_meta_with_values_on_gym_env(sess,
                                        env,
                                        state_ph,
                                        internal_state_ph,
                                        action_ph,
                                        reward_ph,
                                        deterministic_ph,
                                        action_value_op,
                                        action_op,
                                        value_op,
                                        internal_state_op,
                                        zero_state_fn,
                                        initial_action,
                                        num_episodes=1,
                                        deterministic=False,
                                        stream=None,
                                        save_replay=True):
  if stream is None and save_replay:
    raise ValueError('missing `stream` to `save_replay`.')

  rewards = 0.
  for episode in range(num_episodes):
    internal_state = sess.run(zero_state_fn(1, dtype=state_ph.dtype))
    reward = 0.
    action = initial_action
    experiences = []
    next_state = env.reset()
    while True:
      state = next_state
      action_value, action, value, internal_state = sess.run(
          (action_value_op, action_op, value_op, internal_state_op),
          feed_dict={
              state_ph: [[state]],
              **{k: v for k, v in zip(internal_state_ph, internal_state)},
              deterministic_ph: deterministic,
              action_ph: [[action]],
              reward_ph: [[reward]],
          })

      next_state, reward, terminal, _ = env.step(action)
      experiences.append(experience.ExperienceWithValues(
          state, next_state,
          action, action_value,
          value,
          reward, terminal))

      if terminal:
        break
    replay = experience.ReplayWithValues(
        *zip(*experiences),
        sequence_length=len(experiences))
    if save_replay:
      stream.write(replay)
    rewards += sum(replay.reward)
  return rewards


def rollout_meta_on_gym_env(sess,
                            env,
                            state_ph,
                            internal_state_ph,
                            action_ph,
                            reward_ph,
                            deterministic_ph,
                            action_value_op,
                            action_op,
                            internal_state_op,
                            zero_state_fn,
                            initial_action,
                            num_episodes=1,
                            deterministic=False,
                            stream=None,
                            save_replay=True):
  if stream is None and save_replay:
    raise ValueError('missing `stream` to `save_replay`.')

  rewards = 0.
  internal_state = sess.run(zero_state_fn(1, dtype=state_ph.dtype))
  for episode in range(num_episodes):
    # internal_state = sess.run(zero_state_fn(1, dtype=state_ph.dtype))
    reward = 0.
    action = initial_action
    experiences = []
    next_state = env.reset()
    while True:
      state = next_state
      action_value, action, internal_state = sess.run(
          (action_value_op, action_op, internal_state_op),
          feed_dict={
              state_ph: [[state]],
              **{k: v for k, v in zip(internal_state_ph, internal_state)},
              deterministic_ph: deterministic,
              action_ph: [[action]],
              reward_ph: [[reward]],
          })

      next_state, reward, terminal, _ = env.step(action)
      experiences.append(experience.Experience(
          state, next_state,
          action, action_value,
          reward, terminal))

      if terminal:
        break
    replay = experience.Replay(
        *zip(*experiences),
        sequence_length=len(experiences))
    if save_replay:
      stream.write(replay)
    reward_sum = sum(replay.reward)
    print(reward_sum)
    rewards += reward_sum
  return rewards
