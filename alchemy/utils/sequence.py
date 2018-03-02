# -*- coding: utf-8 -*-
import tensorflow as tf

from .utils import ndims


def mask_sequence(lengths, maxlen=None):
  sequence_length_mask = tf.cast(tf.sequence_mask(lengths, maxlen=maxlen), tf.float32)
  sequence_length_total = tf.reduce_sum(sequence_length_mask, axis=-1)
  return sequence_length_mask, sequence_length_total


def list_pad_or_truncate(x, maxlen, pad_value=None):
  """Pad or truncate a list `x` with the values `pad_value` and `maxlen`."""
  length = len(x)
  if maxlen > length:
    x += [pad_value] * (maxlen - length)
  elif maxlen < length:
    x = x[:maxlen]
  return x


def pad_or_truncate(x, maxlen, pad_value=0.):
  """Pad or truncate the second dimension of x by maxlen."""
  rank = ndims(x)
  if rank < 2:
    raise ValueError('pad_or_truncate for x rank = {}d is not defined.')
  xtd = (rank - 2)
  length = tf.shape(x)[1]
  pad = lambda: tf.pad(
      x, [[0, 0], [0, maxlen - length]] + [[0, 0]] * xtd,
      mode="CONSTANT",
      constant_values=pad_value)
  indices = tuple([slice(None), slice(0, maxlen)] + [slice(None)] * xtd)
  truncate = lambda: x[indices]
  return tf.cond(length > maxlen, truncate, pad)


def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  rank = ndims(x)
  if rank < 2:
    raise ValueError('shift_right for x rank = {}d is not defined.')
  xtd = (rank - 2)
  indices = tuple([slice(None), slice(0, -1)] + [slice(None)] * xtd)
  if pad_value is None:
    return tf.pad(
        x, [[0, 0], [1, 0]] + [[0, 0]] * xtd)[indices]
  return tf.concat([pad_value, x], axis=1)[indices]
