# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from alchemy.utils import array_utils


def ndims(x):
  """Return the rank of the tensor as int."""
  return len(x.get_shape())

def normalize(x):
  """Project `x` into the range [0, 1]"""
  return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

def ssd(x, y, extra_dims=2):
  """`Sum Squared over D`: `l2` over `n`-dimensions (starting at `extra_dims`)

  Math:
    ssd(x, y) = sum [l2(x-y)] in range (extra_dims, dims(x|y)]
  """
  assert extra_dims >= 0, "extra_dims must be >= 0, got {}".format(extra_dims)
  shape = y.get_shape().as_list()[extra_dims:]
  return tf.reduce_sum(tf.square(x - y), axis=array_utils.ranged_axes(shape))

def assign_scope(from_scope, to_scope):
  """Return an op that assigns one variable scope to another."""
  assigns = []
  for dst, src in zip(tf.trainable_variables(to_scope), tf.trainable_variables(from_scope)):
    assigns.append(tf.assign(dst, src))
  return tf.group(*assigns)
