# -*- coding: utf-8 -*-
import numpy as np
from random import randint
import tensorflow as tf
from alchemy import utils


class UtilsTest(tf.test.TestCase):

  def test_safe_tf_dtype(self):
    self.assertTrue(utils.safe_tf_dtype(tf.int64) == tf.int64)
    self.assertTrue(utils.safe_tf_dtype(np.array([1]).dtype) == tf.int64)
    self.assertTrue(utils.safe_tf_dtype('int64') == tf.int64)
    self.assertTrue(utils.safe_tf_dtype(int) == tf.int64)

  def test_product(self):
    self.assertTrue(utils.product([1, 2, 3]) == 1 * 2 * 3)

  def test_all_equal(self):
    self.assertTrue(utils.all_equal([1, 2], [1, 2]))

  def test_ndims(self):
    tf.reset_default_graph()
    x = [1, 2, 3]
    tf_x = tf.constant(x)
    self.assertTrue(utils.ndims(tf_x) == 1)

    x = [[1, 2, 3]]
    tf_x = tf.constant(x)
    self.assertTrue(utils.ndims(tf_x) == 2)

  def test_nd_expand_dims(self):
    x = np.array([2])
    expanded_x = utils.nd_expand_dims(x, n=1, before=True)
    self.assertTrue(utils.all_equal(expanded_x.shape, [1, 1]))

    x = np.array([[2, 2]])
    expanded_x = utils.nd_expand_dims(x, n=1, before=True)
    self.assertTrue(utils.all_equal(expanded_x.shape, [1, 1, 2]))

    x = np.array([[2, 2]])
    expanded_x = utils.nd_expand_dims(x, n=1, before=False)
    self.assertTrue(utils.all_equal(expanded_x.shape, [1, 2, 1]))

  def test_list_pad_or_truncate(self):
    x = [1, 2]
    next_x = utils.list_pad_or_truncate(x, 3, 3)
    self.assertTrue(utils.all_equal(next_x, [1, 2] + [3]))

    x = [1, 2, 3]
    next_x = utils.list_pad_or_truncate(x, 2)
    self.assertTrue(utils.all_equal(next_x, [1, 2]))

  def test_pad_or_truncate(self):
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.int32, [1, None])
    with self.test_session() as sess:
      x = np.array([[0, 1, 2]])
      tf_x = utils.pad_or_truncate(x_ph, 4, 3)
      expected_x = np.array([[0, 1, 2, 3]])

      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

      x = np.array([[0, 1, 2, 3, 4]])
      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

  def test_shift_right(self):
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.int32, [1, None])
    with self.test_session() as sess:
      x = np.array([[3, 2, 1]])
      tf_x = utils.shift_right(x_ph)
      expected_x = np.array([[0, 3, 2]])
      actual_x = sess.run(tf_x, feed_dict={x_ph: x})
      self.assertTrue(np.all(np.equal(actual_x, expected_x)))

  def test_normalize(self):
    tf.reset_default_graph()
    x = np.array([0, 1, 2, 3, 4])
    tf_normal_x = utils.normalize(tf.constant(x))
    with self.test_session() as sess:
      normal_x = sess.run(tf_normal_x)
      expected_normal_x = (x - x.min()) / (x.max() - x.min())
      self.assertTrue(utils.eps_equal(normal_x, expected_normal_x, 1e-7))

  def test_mask_sequence(self):
    tf.reset_default_graph()
    inputs_ph = tf.placeholder(tf.float32, [None, None])
    lengths_ph = tf.placeholder(tf.int32, [None])
    sequence_length_mask, sequence_length_total = utils.mask_sequence(lengths_ph, 5)
    masked_inputs = inputs_ph * sequence_length_mask
    masked_means = tf.reduce_sum(masked_inputs, axis=-1) / sequence_length_total

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
  tf.test.main()
