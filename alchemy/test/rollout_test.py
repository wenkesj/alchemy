# -*- coding: utf-8 -*-
import gym

from random import randint

import unittest

from alchemy.memory import rollout


class RolloutTest(unittest.TestCase):

  def test_single_rollout(self):
    env = gym.make('CartPole-v0')

    trajectories = rollout(
        env,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        max_rollout_steps=-1,
        max_trajectory_steps=-1)

    self.assertTrue(len(trajectories) == 1)

  def test_partitioned_rollout(self):
    env = gym.make('CartPole-v0')
    max_rollout_steps = 6
    max_trajectory_steps = 3

    trajectories = rollout(
        env,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        max_rollout_steps=max_rollout_steps,
        max_trajectory_steps=max_trajectory_steps)

    self.assertTrue(len(trajectories) == int(max_rollout_steps / max_trajectory_steps))

if __name__ == '__main__':
  unittest.test.main()
