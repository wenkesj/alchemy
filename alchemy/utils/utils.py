# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf


def safe_tf_dtype(dtype):
  if isinstance(dtype, str):
    return tf.as_dtype(dtype)
  if isinstance(dtype, type):
    return tf.as_dtype(np.dtype(dtype))
  if isinstance(dtype, np.dtype):
    return tf.as_dtype(dtype.name)
  if isinstance(dtype, tf.DType):
    return dtype
  else:
    raise TypeError()


def product(x):
  return np.prod(x)


def ranged_axes(shape):
  axes = (-np.arange(1, len(shape) + 1)[::-1]).tolist()
  if not axes:
    return -1
  return axes


def all_equal(x, y):
  return all([i == j for i, j in zip(x, y)])

def eps_equal(x, y, eps=1e-3):
  return all([abs(i - j) <= eps for i, j in zip(x, y)])


def ndims(x):
  """Return the rank of the tensor as int."""
  return len(x.get_shape())


def nd_expand_dims(x, n=1, before=True):
  if before:
    axes = tuple([np.newaxis] * n + [...])
  else:
    axes = tuple([...] + [np.newaxis] * n)
  return x[axes]


def normalize(x):
  """normalize x -> [0, 1]"""
  return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))


def ssd(x, y):
  shape = y.get_shape().as_list()[2:]
  return tf.reduce_sum(tf.square(x - y), axis=ranged_axes(shape))


def group_scope(scope):
  """Return an op that groups a collection by scope."""
  return tf.trainable_variables(scope)


def assign_scope(from_scope, to_scope):
  """Return an op that assigns one variable scope to another."""
  assigns = []
  for dst, src in zip(group_scope(to_scope), group_scope(from_scope)):
    assigns.append(tf.assign(dst, src))
  return tf.group(*assigns)


def partition(zipped, num_steps):
  size = len(zipped)
  parts = []
  for i in range(0, size, num_steps):
    end = i + num_steps
    if end >= size:
      parts.append(zip(*zipped[i:]))
      break
    else:
      parts.append(zip(*zipped[i:end]))
  return parts
