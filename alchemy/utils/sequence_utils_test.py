# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from alchemy.utils import sequence_utils


class SequenceUtilsTest(test.TestCase):

  def test_pad_or_truncate(self):
    ops.reset_default_graph()
    x_ph = array_ops.placeholder(dtypes.int32, [1, None])
    with self.test_session() as sess:
      x = np.array([[0, 1, 2]])
      tf_x = sequence_utils.pad_or_truncate(x_ph, maxsize=4, axis=1, pad_value=3)
      expected_x = np.array([[0, 1, 2, 3]])

      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

      x = np.array([[0, 1, 2, 3, 4]])
      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

  def test_shift_right(self):
    ops.reset_default_graph()
    x_ph = array_ops.placeholder(dtypes.int32, [1, None])
    with self.test_session() as sess:
      x = np.array([[3, 2, 1]])
      tf_x = sequence_utils.shift_right(x_ph, axis=1, rotations=1)
      expected_x = np.array([[0, 3, 2]])
      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

  def test_mask_sequence(self):
    ops.reset_default_graph()
    inputs_ph = array_ops.placeholder(dtypes.float32, [None, None])
    lengths_ph = array_ops.placeholder(dtypes.int32, [None])
    sequence_length_mask, sequence_length_total = sequence_utils.mask_sequence(lengths_ph, 5)
    masked_inputs = inputs_ph * sequence_length_mask
    masked_means = math_ops.reduce_sum(masked_inputs, axis=-1) / sequence_length_total

    with self.test_session() as sess:
      x = np.array([[0., 1., 2., 3., 4.]])
      lengths = np.array([3])

      expected_masked_inputs = np.array([[0., 1., 2., 0., 0.]])
      expected_masked_means = np.array([1])

      actual_masked_inputs, actual_masked_means = sess.run(
          (masked_inputs, masked_means),
          feed_dict={
            inputs_ph: x,
            lengths_ph: lengths,
          })
      self.assertTrue(np.all(np.equal(expected_masked_inputs, actual_masked_inputs)))
      self.assertTrue(np.all(np.equal(expected_masked_means, actual_masked_means)))


if __name__ == '__main__':
  test.main()
