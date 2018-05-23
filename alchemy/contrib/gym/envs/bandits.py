# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from gym.core import Env
from gym.spaces import discrete
from gym.utils import seeding

from alchemy.utils import assert_utils


class BanditEnv(Env):

  """
  `BanditEnv` is a `gym.Env` implements the classic 2-bandit problem with either
  independent or dependent arms.

  Further apart the probabilities are, the easier it is to determine the rewards.
  """

  def __init__(self, probs, max_time=99):
    self.time = 0
    self.max_time = max_time
    # independent case where probs are independent of each other
    # where p = p1 + p2, p >= 1 and > 0
    if isinstance(probs, list) or isinstance(probs, tuple):
      assert_utils.assert_true(
          len(probs) == 2,
          "if `probs` was meant to be a `list`/`tuple`, then it must be of size 2.")
      self.bandit = np.array([probs[0], probs[1]])
    # dependent bandits
    # where p = p1 + (1 - p1), p = 1
    else:
      self.bandit = np.array([probs, 1. - probs])

    self.action_space = discrete.Discrete(len(self.bandit))
    self.observation_space = discrete.Discrete(1)
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert_utils.assert_true(
        self.action_space.contains(action),
        '`action_space` must contain `action`.')
    # sample from bernoulli distribution with p = bandit[action]
    reward = int(np.random.uniform() < self.bandit[action])
    self.time += 1
    done = self.time > self.max_time
    return [self.time], reward, done, {}

  def reset(self):
    self.time = 0
    return [self.time]

  def render(self, mode='human', close=False):
    pass


class ConstantBanditEnv(Env):

  """
  `ConstantBanditEnv` is a `gym.Env` implements deterministic, 11-bandit problem with informative
  rewards.
  """

  def __init__(self, max_time=5):
    self.time = 0
    self.max_time = max_time
    assert_utils.assert_true(
        len(probs) == 11,
        "if `probs` was meant to be a `list`/`tuple`, then it must be of size 2.")
    self.bandit = np.array([1.] * 11)
    self.informative_action = 0
    self.action_space = discrete.Discrete(len(self.bandit))
    self.observation_space = discrete.Discrete(1)
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert_utils.assert_true(
        self.action_space.contains(action),
        '`action_space` must contain `action`.')

    if self.time == 0:
      if action == self.informative_action:
        reward = .55
      else:
        reward = 1.4
    else:
      reward = self.bandit[action]

    self.time += 1
    done = self.time > self.max_time
    return [self.time], reward, done, {}

  def reset(self):
    self.informative_action = np.random.choice(range(11))
    self.bandit = np.array([1.] * 11)
    self.bandit[11] = self.informative_action / 10.
    self.bandit[self.informative_action] = 5.
    self.time = 0
    return [self.time]

  def render(self, mode='human', close=False):
    pass


class BanditUniform(BanditEnv):
  def __init__(self):
    super(BanditUniform, self).__init__(np.random.uniform())


class BanditEasy(BanditEnv):
  def __init__(self):
    super(BanditEasy, self).__init__(np.random.uniform(.1, .9))


class BanditMedium(BanditEnv):
  def __init__(self):
    super(BanditMedium, self).__init__(np.random.uniform(.25, .75))


class BanditHard(BanditEnv):
  def __init__(self):
    super(BanditHard, self).__init__(np.random.uniform(.4, .6))
