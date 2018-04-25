# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from alchemy.contrib.layers import autoencoder
from alchemy.utils import array_utils


class AutoEncoderTest(test.TestCase):

  def test_conv2d_autoencoder(self):
    ops.reset_default_graph()
    inputs_ph = array_ops.placeholder(dtypes.float32, [None, 8, 8, 1])

    strides = [1, 1]
    latent_output, encoder, shapes = autoencoder.conv2d_encoder(
        inputs_ph, [2, 2], [2, 2], strides, 'encoder')

    outputs = autoencoder.conv2d_decoder(
        latent_output, encoder, shapes, strides, 'decoder')

    self.assertTrue(
        array_utils.all_equal(
            outputs.get_shape().as_list(),
            inputs_ph.get_shape().as_list()))

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      input_image = np.zeros((1, 8, 8, 1))
      output_image = sess.run(outputs, feed_dict={inputs_ph: input_image})
      self.assertTrue(np.all(np.equal(input_image, output_image)))


if __name__ == '__main__':
  test.main()
