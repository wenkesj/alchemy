# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import numbers


def safe_tf_dtype(dtype):
  if isinstance(dtype, str):
    return tf.as_dtype(dtype)
  if isinstance(dtype, np.dtype):
    return tf.as_dtype(dtype)
  if isinstance(dtype, tf.DType):
    return dtype
  if issubclass(dtype, numbers.Number):
    return tf.as_dtype(np.dtype(dtype))
  raise TypeError('Cannot safely assume type: {}'.format(dtype))
