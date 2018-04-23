# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import unittest

from alchemy.utils import array_utils


class ArrayUtilsTest(unittest.TestCase):

  def test_product(self):
    self.assertTrue(array_utils.product([1, 2, 3]) == 1 * 2 * 3)

  def test_all_equal(self):
    self.assertTrue(array_utils.all_equal([1, 2], [1, 2]))

  def test_nd_expand_dims(self):
    x = np.array([2])
    expanded_x = array_utils.nd_expand_dims(x, n=1, before=True)
    self.assertTrue(array_utils.all_equal(expanded_x.shape, [1, 1]))

    x = np.array([[2, 2]])
    expanded_x = array_utils.nd_expand_dims(x, n=1, before=True)
    self.assertTrue(array_utils.all_equal(expanded_x.shape, [1, 1, 2]))

    x = np.array([[2, 2]])
    expanded_x = array_utils.nd_expand_dims(x, n=1, before=False)
    self.assertTrue(array_utils.all_equal(expanded_x.shape, [1, 2, 1]))

  def test_list_pad_or_truncate(self):
    x = [1, 2]
    next_x = array_utils.list_pad_or_truncate(x, 3, 3)
    self.assertTrue(array_utils.all_equal(next_x, [1, 2] + [3]))

    x = [1, 2, 3]
    next_x = array_utils.list_pad_or_truncate(x, 2)
    self.assertTrue(array_utils.all_equal(next_x, [1, 2]))


if __name__ == '__main__':
  unittest.main()
