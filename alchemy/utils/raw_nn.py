# -*- coding: utf-8 -*-
import numpy as np


def relu(x):
  return np.maximum(0., x)


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x)
