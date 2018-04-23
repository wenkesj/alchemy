# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import deque
import gym
from random import randint
import tensorflow as tf
import time

from alchemy.memory import memory
from alchemy.memory import rollouts


class CartPoleReplayMemory(memory.Memory):
  def __init__(self):
    self.ram = deque([])

  @property
  def state_dtype(self):
    return float

  @property
  def state_shape(self):
    return [4]

  @property
  def action_dtype(self):
    return int

  @property
  def action_shape(self):
    return []

  @property
  def action_value_dtype(self):
    return float

  @property
  def action_value_shape(self):
    return [2]

  def write(self, mem):
    self.ram.append(mem)

  def read(self):
    return self.ram.popleft()

  def clear(self):
    return self.ram.clear()

  def __len__(self):
    return len(self.ram)


class RolloutTest(tf.test.TestCase):

  def test_single_rollout(self):
    env = gym.make('CartPole-v0')

    trajectories = rollouts.rollout(
        env,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        max_rollout_steps=-1,
        max_trajectory_steps=-1)

    self.assertTrue(len(trajectories) == 1)

  def test_partitioned_rollout(self):
    env = gym.make('CartPole-v0')
    max_rollout_steps = 6
    max_trajectory_steps = 3

    trajectories = rollouts.rollout(
        env,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        max_rollout_steps=max_rollout_steps,
        max_trajectory_steps=max_trajectory_steps)

    self.assertTrue(len(trajectories) == int(max_rollout_steps / max_trajectory_steps))

  def test_rollout_to_src(self):
    ram = CartPoleReplayMemory()
    env = gym.make('CartPole-v0')

    num_episodes = 1
    max_rollout_steps = 6
    max_trajectory_steps = 3

    rollouts.rollout_to_src(ram, env,
                   step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
                   num_episodes=num_episodes,
                   max_rollout_steps=max_rollout_steps,
                   max_trajectory_steps=max_trajectory_steps)
    self.assertTrue(len(ram) == num_episodes * int(max_rollout_steps / max_trajectory_steps))
    ram.clear()

  def test_sync_pool(self):
    ram = CartPoleReplayMemory()

    num_envs = 2
    num_episodes = 2
    max_rollout_steps = 6
    max_trajectory_steps = 3

    rollouts.RolloutPool(
        lambda: gym.make('CartPole-v0'),
        num_envs=num_envs,
        num_threads=1)(
            ram,
            step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
            synchronous=True,
            num_episodes=num_episodes,
            max_rollout_steps=max_rollout_steps,
            max_trajectory_steps=max_trajectory_steps)

    self.assertTrue(
        len(ram) == num_envs * num_episodes * int(max_rollout_steps / max_trajectory_steps))
    ram.clear()

  def test_async_pool(self):
    ram = CartPoleReplayMemory()

    num_envs = 2
    num_episodes = 2
    max_rollout_steps = 6
    max_trajectory_steps = 3

    pool = rollouts.RolloutPool(
        lambda: gym.make('CartPole-v0'),
        num_envs=num_envs,
        num_threads=2)
    pool(
        ram,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        synchronous=False,
        num_episodes=num_episodes,
        max_rollout_steps=max_rollout_steps,
        max_trajectory_steps=max_trajectory_steps)

    time.sleep(1)

    while True:
      if pool.done:
        break

    self.assertTrue(
        len(ram) == num_envs * num_episodes * int(max_rollout_steps / max_trajectory_steps))
    ram.clear()

  def test_rollout_dataset(self):
    tf.reset_default_graph()
    ram = CartPoleReplayMemory()

    num_envs = 2
    num_episodes = 2
    max_rollout_steps = 6
    max_trajectory_steps = max_sequence_length = 3

    pool = rollouts.RolloutPool(
        lambda: gym.make('CartPole-v0'),
        num_envs=num_envs,
        num_threads=1)
    pool(
        ram,
        step_fn=lambda state, internals: (randint(0, 1), [0., 0.], None),
        synchronous=True,
        num_episodes=num_episodes,
        max_rollout_steps=max_rollout_steps,
        max_trajectory_steps=max_trajectory_steps)

    batch_size = num_envs * num_episodes
    dataset = rollouts.rollout_dataset(
        ram,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length)

    with self.test_session() as sess:
      batch = (state, next_state, action, value, reward, terminal, sequence_length) = sess.run(
          dataset)
      for feature in batch[:-1]:
        self.assertTrue(feature.shape[0], batch_size)
        self.assertTrue(feature.shape[1], max_sequence_length)



if __name__ == '__main__':
  tf.test.main()
