# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from alchemy.utils import array_utils
from alchemy.utils import shortcuts


class ShortcutsTest(test.TestCase):

  def test_normalize(self):
    ops.reset_default_graph()
    x = np.array([0, 1, 2, 3, 4])
    tf_normal_x = shortcuts.normalize(constant_op.constant(x))

    with self.test_session() as sess:
      normal_x = sess.run(tf_normal_x)
      expected_normal_x = (x - x.min()) / (x.max() - x.min())
      self.assertTrue(array_utils.all_equal(normal_x, expected_normal_x, eps=1e-7))

  def test_ndims(self):
    ops.reset_default_graph()
    x = [1, 2, 3]
    tf_x = constant_op.constant(x)
    self.assertTrue(shortcuts.ndims(tf_x) == 1)

    x = [[1, 2, 3]]
    tf_x = constant_op.constant(x)
    self.assertTrue(shortcuts.ndims(tf_x) == 2)


if __name__ == '__main__':
  test.main()
