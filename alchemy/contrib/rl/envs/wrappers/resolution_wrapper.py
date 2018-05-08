# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gym
import numpy as np
import scipy


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
