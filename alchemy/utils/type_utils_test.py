# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import unittest

from tensorflow.python.framework import dtypes

from alchemy.utils import type_utils


class TypeUtilsTest(unittest.TestCase):

  def test_safe_tf_dtype(self):
    self.assertTrue(type_utils.safe_tf_dtype(dtypes.int64) == dtypes.int64)
    self.assertTrue(type_utils.safe_tf_dtype(np.array([1]).dtype) == dtypes.int64)
    self.assertTrue(type_utils.safe_tf_dtype('int64') == dtypes.int64)
    self.assertTrue(type_utils.safe_tf_dtype(int) == dtypes.int64)


if __name__ == '__main__':
  unittest.main()
