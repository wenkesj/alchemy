# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf

import unittest

from alchemy import spaces


class SpaceTest(unittest.TestCase):

  def test_categorical_space(self):
    tf.reset_default_graph()
    input_shape = [8,]
    inputs_ph = tf.placeholder(tf.float32, [None,] + input_shape)

    space = spaces.CategoricalSpace(input_shape, is_one_hot=True)

    inputs_sample = space.build_mode_op(inputs_ph)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      inputs = np.random.uniform(low=-10, high=10, size=[1,] + input_shape)
      outputs = sess.run(inputs_sample, feed_dict={inputs_ph: inputs})
      self.assertTrue(
          np.all(np.equal(np.identity(inputs.shape[-1])[np.argmax(inputs)], outputs)))

  def test_continuous_space(self):
    # TODO
    tf.reset_default_graph()
    input_shape = [2,]
    inputs_ph = tf.placeholder(tf.float32, [None,] + input_shape)

    low, high = np.array([-1., 0.]), np.array([0., 1.])
    space = spaces.ContinuousSpace(low, high, input_shape)

    inputs_sample = space.build_sample_op(inputs_ph)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      inputs = np.random.uniform(low=[-10, -1.], high=[10., 0.], size=[1,] + input_shape)
      outputs = sess.run(inputs_sample, feed_dict={inputs_ph: inputs})


if __name__ == '__main__':
  tf.test.main()
