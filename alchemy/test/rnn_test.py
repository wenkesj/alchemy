# -*- coding: utf-8 -*-
import numpy as np

import tensorflow as tf

import unittest

from alchemy import layers, utils


class RNNTest(unittest.TestCase):

  def test_stacked_rnn(self):
    cell_fns = [
      lambda size: tf.contrib.rnn.BasicRNNCell(size),
      lambda size: tf.contrib.rnn.BasicLSTMCell(size),
      lambda size: tf.contrib.rnn.LSTMCell(size),
      lambda size: tf.contrib.rnn.GRUCell(size),
    ]

    for cell_fn in cell_fns:
      tf.reset_default_graph()
      dtype = tf.float32
      inputs_ph = tf.placeholder(dtype, [None, None, 1])

      scope = 'stacked_rnn'
      outputs, states, initial_state_phs, zero_states = layers.stacked_rnn(
          inputs_ph, [3, 2], cell_fn, scope)

      batch_size = 1
      num_iters = 4
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        initial_states = sess.run(zero_states(batch_size, dtype))
        for _ in range(num_iters):
          input_seq = np.zeros((batch_size, 1, 1))
          output_seq, initial_states = sess.run(
              (outputs, states),
              feed_dict={
                inputs_ph: input_seq,
                **{k: v for k, v in zip(initial_state_phs, initial_states)},
              })
