# -*- coding: utf-8 -*-
from __future__ import absolute_import

from alchemy.contrib.rl import experience


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
