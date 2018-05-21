# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from gym.core import Env
from gym.spaces import discrete
from gym.utils import seeding


class BanditEnv(Env):
  def __init__(self, probs, max_time=99):
    self.time = 0
    self.max_time = max_time
    self.bandit = np.array([probs, 1 - probs])
    self.action_space = discrete.Discrete(len(self.bandit))
    self.observation_space = discrete.Discrete(1)
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action)
    reward = int(np.random.uniform() < self.bandit[action])
    done = self.time > self.max_time

    self.time += 1
    return [self.time], reward, done, {}

  def reset(self):
    self.time = 0
    return [self.time]

  def render(self, mode='human', close=False):
    pass


class BanditUniform(BanditEnv):
  def __init__(self):
    BanditEnv.__init__(self, np.random.uniform())


class BanditEasy(BanditEnv):
  def __init__(self):
    BanditEnv.__init__(self, np.random.choice([0.9, 0.1]))


class BanditMedium(BanditEnv):
  def __init__(self):
    BanditEnv.__init__(self, np.random.choice([0.75, 0.25]))


class BanditHard(BanditEnv):
  def __init__(self):
    BanditEnv.__init__(self, np.random.choice([0.6, 0.4]))
