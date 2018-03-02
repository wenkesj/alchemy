from __future__ import print_function

import gym

import itertools

import numpy as np

from random import choice

import scipy

from time import sleep

from vizdoom import DoomGame, ScreenFormat, ScreenResolution, Mode


class VizDoomEnv(gym.Env):

  def __init__(self, config='my_way_home.cfg', repeat_action=1, render=False):
    self._game = DoomGame()
    self._game.load_config(config)
    self._game.set_mode(Mode.PLAYER)
    self._game.set_screen_format(ScreenFormat.GRAY8)
    self._game.set_screen_resolution(ScreenResolution.RES_640X480)
    self._game.set_window_visible(render)
    self._game.init()
    self._actions = self._get_actions()
    self._repeat_action = repeat_action
    self._is_rendered = False

  def _get_actions(self):
    num_actions = self._game.get_available_buttons_size()
    actions = []
    for perm in itertools.product([False, True], repeat=num_actions):
      actions.append(list(perm))
    return actions

  def _get_observation(self):
    state = self._game.get_state()
    if state is not None:
      return state.screen_buffer
    return None

  def _get_terminal(self):
    return self._game.is_episode_finished()

  def reset(self):
    self._game.new_episode()
    return self._get_observation()

  def step(self, action):
    action_ = self._actions[action]
    reward = self._game.make_action(action_, self._repeat_action)
    return self._get_observation(), reward, self._get_terminal(), []

  def render(self, mode='human'):
    self._game.set_window_visible(True)

  def close(self):
    self._game.close()


class ResolutionWrapper(gym.ObservationWrapper):
  def __init__(self, env, resolution):
    super(ResolutionWrapper, self).__init__(env)
    self._resolution = resolution

  @staticmethod
  def resize_image(img, resolution):
    if img is None:
      return np.zeros(resolution + [1,], dtype=np.float32)
    img = scipy.misc.imresize(img, resolution)
    img = img.astype(np.float32) / 126.
    return np.expand_dims(img, -1)

  def observation(self, img):
    return ResolutionWrapper.resize_image(img, self._resolution)


if __name__ == '__main__':
  env = ResolutionWrapper(VizDoomEnv('my_way_home.cfg', 5, True), (30, 45))

  state = env.reset()
  env.render()

  while True:
    sleep(0.028)
    state, reward, terminal, info = env.step(choice(range(5)))
    if terminal:
      break
  env.close()
