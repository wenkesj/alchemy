# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from alchemy.utils import shortcuts


def mask_sequence(lengths, maxlen):
  """Returns (mask tensor according to the max of `lengths`, tensor length of each sequence)"""
  sequence_length_mask = tf.cast(tf.sequence_mask(lengths, maxlen=maxlen), tf.float32)
  sequence_length_total = tf.reduce_sum(sequence_length_mask, axis=-1)
  return sequence_length_mask, sequence_length_total

def pad_or_truncate(x, maxsize, axis=-1, pad_value=0):
  """Pad or truncate the dimension according to `axis` of x by `maxsize`, with `pad_value`."""
  rank = shortcuts.ndims(x)
  size = tf.shape(x)[axis]
  value_padding = [[0, 0]] * rank
  value_padding[axis] = [0, maxsize - size]

  # pad op
  pad = lambda: tf.pad(
      x, value_padding,
      mode="CONSTANT",
      constant_values=pad_value)
  index_padding = [slice(None)] * rank
  index_padding[axis] = slice(0, maxsize)
  index_padding = tuple(index_padding)

  # truncate op
  truncate = lambda: x[index_padding]

  return tf.cond(size > maxsize, truncate, pad)

def shift_right(x, axis=1, rotations=1, pad_value=None):
  """Shift the dimension according to `axis` of `x` right by `rotations`."""
  rank = shortcuts.ndims(x)
  index_padding = [slice(None)] * rank
  index_padding[axis] = slice(0, -1)
  index_padding = tuple(index_padding)

  if pad_value is None:
    value_padding = [[0, 0]] * rank
    value_padding[axis] = [rotations, 0]
    return tf.pad(x, value_padding)[index_padding]

  return tf.concat([pad_value, x], axis=axis)[index_padding]
