# -*- coding: utf-8 -*-
from __future__ import absolute_import

import functools
import numpy as np

from tensorflow.python.ops import gen_array_ops


def all_equal(x, y, eps=None):
  """Return if x == y, if eps is not None, return if abs(x-y) <= eps"""
  if eps:
    return all([abs(i - j) <= eps
                for i, j in zip(x, y)
                if i is not None and j is not None])
  return all([i == j for i, j in zip(x, y)])

def product(x):
  """Reduce product of x."""
  return functools.reduce(lambda x, y: x * y, x)

def flatten(x):
  return gen_array_ops.reshape(x, [product(x.get_shape().as_list())])

def unflatten(x, shapes):
  arrays = []
  start = 0
  for shape in shapes:
    end = product(shape)
    arrays.append(gen_array_ops.reshape(x[start:start+end], shape))
    start += end
  return arrays

def nd_flatten(x):
  return x.ravel()

def nd_unflatten(x, shapes):
  arrays = []
  start = 0
  for shape in shapes:
    end = product(shape)
    arrays.append(x[start:start+end].reshape(shape))
    start += end
  return arrays

def nd_expand_dims(x, n=1, before=True):
  """Expand multiple dimensions, i.e. add 1 after or before

  Note:
    x = np.reshape(x, [1, 1, 1, 1, 1, 6, 9])
      = nd_expand_dims(x, n=5)
    -or-
    x = np.reshape(x, [6, 9, 1, 1, 1, 1, 1])
      = nd_expand_dims(x, n=5, before=False)
  """
  if before:
    axes = tuple([np.newaxis] * n + [...])
  else:
    axes = tuple([...] + [np.newaxis] * n)
  return x[axes]

def ranged_axes(shape):
  """Return a `list` of `int` that represents a range of axes."""
  return (-np.arange(1, len(shape) + 1)[::-1]).tolist() or -1

def partition(zipped, num_steps, allow_overflow=True):
  """Partition `zipped` into `num_steps`.

  Note:
    if num_steps is not divisible, the rest is added if `allow_overflow`.
  """
  size = len(zipped)
  parts = []
  for i in range(0, size, num_steps):
    end = i + num_steps
    if end >= size:
      parts.append(zip(*zipped[i:]))
      break
    elif allow_overflow:
      parts.append(zip(*zipped[i:end]))
  return parts

def list_pad_or_truncate(x, maxlen, pad_value=None):
  """Pad or truncate a list `x` with the values `pad_value` and `maxlen`."""
  length = len(x)
  if maxlen > length:
    x += [pad_value] * (maxlen - length)
  elif maxlen < length:
    x = x[:maxlen]
  return x
