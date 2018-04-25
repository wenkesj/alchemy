# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.client import session
from tensorflow.contrib import rnn

from alchemy.contrib.rnn import stacked_rnn_impl


class RNNTest(test.TestCase):

  def test_stacked_rnn(self):
    cell_fns = [
      lambda size: rnn.BasicRNNCell(size),
      lambda size: rnn.BasicLSTMCell(size),
      lambda size: rnn.LSTMCell(size),
      lambda size: rnn.GRUCell(size),
    ]

    for cell_fn in cell_fns:
      ops.reset_default_graph()
      dtype = dtypes.float32
      inputs_ph = array_ops.placeholder(dtype, [None, None, 1])

      scope = 'stacked_rnn'
      outputs, states, initial_state_phs, zero_states = stacked_rnn_impl.stacked_rnn(
          inputs_ph, [2, 4], cell_fn, scope)

      batch_size = 1
      num_iters = 4
      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())

        initial_states = sess.run(zero_states(batch_size, dtype))
        for _ in range(num_iters):
          input_seq = np.zeros((batch_size, 1, 1))
          output_seq, initial_states = sess.run(
              (outputs, states),
              feed_dict={
                inputs_ph: input_seq,
                **{k: v for k, v in zip(initial_state_phs, initial_states)},
              })


if __name__ == '__main__':
  test.main()
