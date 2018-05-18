# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
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

def normalize(x, axis=-1):
  """Project `x` into the range [0, 1]"""
  min_x = array_ops.expand_dims(math_ops.reduce_min(x, axis=axis), axis)
  return (x - min_x) / (array_ops.expand_dims(math_ops.reduce_max(x, axis=axis), axis) - min_x)

def batch_norm(op):
  mean_op = array_ops.expand_dims(
      math_ops.reduce_mean(
          op, axis=0), 0)

  diff_op = op - mean_op
  stdv_op = math_ops.sqrt(
      array_ops.expand_dims(
          math_ops.reduce_mean(
              math_ops.square(diff_op), axis=0), 0))
  return ((op - mean_op) + distribution_utils.epsilon) / (
      stdv_op + distribution_utils.epsilon)

def cummean(op, length, max_length):
  mask = math_ops.cast(array_ops.sequence_mask(length, maxlen=max_length), op.dtype)
  length_expanded = array_ops.expand_dims(length, -1)
  mean_op = math_ops.cumsum(
      op, axis=-1, reverse=False) / math_ops.cast(
          length_expanded, op.dtype)
  return mean_op * mask

def cumstdv(op, mean_op, length, max_length):
  mask = math_ops.cast(array_ops.sequence_mask(length, maxlen=max_length), op.dtype)
  length_expanded = array_ops.expand_dims(length, -1)
  stdv_op = math_ops.sqrt(
      math_ops.cumsum(
          op - mean_op, axis=-1, reverse=False) / math_ops.cast(
              length_expanded, op.dtype))
  return stdv_op * mask

def cumstandardize(op, length, max_length):
  mean_op = cummean(op, length, max_length)
  stdv_op = cumstdv(op, mean_op, length, max_length)
  return ((op - mean_op) + distribution_utils.epsilon) / (
      stdv_op + distribution_utils.epsilon) * mask

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
