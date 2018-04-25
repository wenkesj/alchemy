# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import numbers

from tensorflow.python.framework import dtypes


def safe_tf_dtype(dtype):
  if isinstance(dtype, str):
    return dtypes.as_dtype(dtype)
  if isinstance(dtype, np.dtype):
    return dtypes.as_dtype(dtype)
  if isinstance(dtype, dtypes.DType):
    return dtype
  if issubclass(dtype, numbers.Number):
    return dtypes.as_dtype(np.dtype(dtype))
  raise TypeError('Cannot safely assume type: {}'.format(dtype))
