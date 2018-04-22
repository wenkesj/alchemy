# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf

from alchemy import layers, utils


class AutoEncoderTest(tf.test.TestCase):

  def test_conv2d_autoencoder(self):
    tf.reset_default_graph()
    inputs_ph = tf.placeholder(tf.float32, [None, 8, 8, 1])

    scope = 'autoencoder'
    strides = [1, 1]
    latent_output, encoder, shapes = layers.conv2d_encoder(
        inputs_ph, [2, 2], [2, 2], strides, 'encoder')

    outputs = layers.conv2d_decoder(
        latent_output, encoder, shapes, strides, 'decoder')

    self.assertTrue(
        utils.all_equal(
            outputs.get_shape().as_list(),
            inputs_ph.get_shape().as_list()))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      input_image = np.zeros((1, 8, 8, 1))
      output_image = sess.run(outputs, feed_dict={inputs_ph: input_image})
      self.assertTrue(np.all(np.equal(input_image, output_image)))

if __name__ == '__main__':
  tf.test.main()
