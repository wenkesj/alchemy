# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops

from alchemy.utils import shortcuts

# TODO(wenkesj): clean this up, this isn't good -_-
def gather_along_second_axis(data, indices):
  ndims = len(data.get_shape().as_list())
  shape = array_ops.shape(data)
  re_shape = [shape[0] * shape[1]]
  indices = array_ops.reshape(indices, re_shape)
  for idx in range(2, ndims):
    re_shape.append(shape[idx])
  data = array_ops.reshape(data, re_shape)
  batch_offset = math_ops.range(0, array_ops.shape(data)[0])
  flat_indices = array_ops.stack([batch_offset, indices], axis=1)
  two_d = gen_array_ops.gather_nd(data, flat_indices)

  if ndims == 4:
    three_d = gen_array_ops.reshape(two_d, [shape[0], shape[1], -1])
  elif ndims == 3:
    three_d = gen_array_ops.reshape(two_d, [shape[0], shape[1]])
  return three_d

def expand_dims(x, axes):
  for axis in axes:
    x = array_ops.expand_dims(x, axis=axis)
  return x

def mask_sequence(lengths, maxlen, dtype=dtypes.float32):
  """Returns (mask tensor according to the max of `lengths`, tensor length of each sequence)"""
  sequence_length_mask = math_ops.cast(array_ops.sequence_mask(lengths, maxlen=maxlen), dtype)
  sequence_length_total = math_ops.reduce_sum(sequence_length_mask, axis=-1)
  return sequence_length_mask, sequence_length_total

def pad_or_truncate(x, maxsize, axis=-1, pad_value=0):
  """Pad or truncate the dimension according to `axis` of x by `maxsize`, with `pad_value`."""
  rank = shortcuts.ndims(x)
  size = array_ops.shape(x)[axis]
  value_padding = [[0, 0]] * rank
  value_padding[axis] = [0, maxsize - size]

  # pad op
  pad = lambda: array_ops.pad(
      x, value_padding,
      mode="CONSTANT",
      constant_values=pad_value)
  index_padding = [slice(None)] * rank
  index_padding[axis] = slice(0, maxsize)
  index_padding = tuple(index_padding)

  # truncate op
  truncate = lambda: x[index_padding]

  return control_flow_ops.cond(size > maxsize, truncate, pad)

def shift(x, axis=1, rotations=1, pad_value=None):
  """Shift the dimension according to `axis` of `x` right by `rotations`."""
  direction = abs(rotations)
  is_right = direction == rotations

  rank = shortcuts.ndims(x)
  index_padding = [slice(None)] * rank
  index_padding[axis] = slice(0, -1) if is_right else slice(1, None)
  index_padding = tuple(index_padding)

  if pad_value is None:
    value_padding = [[0, 0]] * rank
    value_padding[axis] = [direction, 0] if is_right else [0, direction]
    return array_ops.pad(x, value_padding)[index_padding]

  padded = [pad_value, x] if is_right else [x, pad_value]
  return array_ops.concat(padded, axis=axis)[index_padding]
