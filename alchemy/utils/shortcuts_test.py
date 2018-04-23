# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from alchemy.utils import array_utils
from alchemy.utils import shortcuts


class ShortcutsTest(tf.test.TestCase):

  def test_normalize(self):
    tf.reset_default_graph()
    x = np.array([0, 1, 2, 3, 4])
    tf_normal_x = shortcuts.normalize(tf.constant(x))

    with self.test_session() as sess:
      normal_x = sess.run(tf_normal_x)
      expected_normal_x = (x - x.min()) / (x.max() - x.min())
      self.assertTrue(array_utils.all_equal(normal_x, expected_normal_x, eps=1e-7))

  def test_ndims(self):
    tf.reset_default_graph()
    x = [1, 2, 3]
    tf_x = tf.constant(x)
    self.assertTrue(shortcuts.ndims(tf_x) == 1)

    x = [[1, 2, 3]]
    tf_x = tf.constant(x)
    self.assertTrue(shortcuts.ndims(tf_x) == 2)


if __name__ == '__main__':
  tf.test.main()
