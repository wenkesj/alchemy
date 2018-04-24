# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test

from alchemy.contrib.rnn import fast_weights_impl


class FastWeightsTest(test.TestCase):

  def test_FastWeightsRNNCell(self):
    with self.test_session():
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inputs = array_ops.zeros([1, 2])
        initial_states = (array_ops.zeros([1, 2]), array_ops.zeros([1, 2, 2]))
        cell = fast_weights_impl.FastWeightsRNNCell(2)
        outputs, states = cell(inputs, initial_states)
        variables.global_variables_initializer().run()
        self.assertTrue(np.all(np.equal(outputs.eval(), np.array([[0., 0.]]))))


if __name__ == '__main__':
  test.main()
