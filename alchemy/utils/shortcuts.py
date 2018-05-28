# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables

from alchemy.utils import array_utils
from alchemy.utils import distribution_utils
from alchemy.utils import assert_utils


def placeholder_like(ph, name=None):
  return array_ops.placeholder(ph.dtype, ph.shape, name=name)

def ndims(x):
  """Return the rank of the tensor as int."""
  return len(x.get_shape())

def zero_to_one(x, axis=-1):
  """Project `x` into the range [0, 1]"""
  min_x = array_ops.expand_dims(math_ops.reduce_min(x, axis=axis), axis)
  return (x - min_x) / (array_ops.expand_dims(math_ops.reduce_max(x, axis=axis), axis) - min_x)

def cummean(op, length, max_length):
  mask = math_ops.cast(array_ops.sequence_mask(length, maxlen=max_length), op.dtype)
  length_expanded = array_ops.expand_dims(length, -1)
  mean_op = math_ops.cumsum(
      op, axis=-1, reverse=False) / math_ops.cast(
          length_expanded, op.dtype)
  return mean_op * mask

def cumnormalize(op, length, max_length, scale=True, center=True):
  if center:
    mean = cummean(
        op, length, max_length)
    op -= mean
  if scale:
    variance = math_ops.sqrt(cummean(math_ops.square(op), length, max_length))
    op /= array_ops.where(
        gen_math_ops.equal(variance, 0),
        array_ops.ones_like(variance),
        variance)
  return op

def ssd(x, y, extra_dims=2):
  """`Sum Squared over D`: `l2` over `n`-dimensions (starting at `extra_dims`)

  Math:
    ssd(x, y) = sum [l2(x-y)] in range (extra_dims, dims(x|y)]
  """
  assert_utils.assert_true(extra_dims >= 0, "extra_dims must be >= 0, got {}".format(extra_dims))
  shape = y.get_shape().as_list()[extra_dims:]
  return math_ops.reduce_sum(math_ops.square(x - y), axis=array_utils.ranged_axes(shape))

def assign_scope(from_scope, to_scope):
  """Return an op that assigns one variable scope to another."""
  assigns = []
  to_vars = variables.trainable_variables(to_scope)
  from_vars = variables.trainable_variables(from_scope)
  for dst, src in zip(to_vars, from_vars):
    assigns.append(state_ops.assign(dst, src))
  return control_flow_ops.group(*assigns)
