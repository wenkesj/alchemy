# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

from alchemy.contrib.layers import wta_impl


class WTATest(test.TestCase):

  def test_call_wta(self):
    with self.test_session():
      layer = wta_impl.WTA(1, name='wta')
      inputs = constant_op.constant([0., .5, 1., .5, .0])
      outputs = layer(inputs)
      outputs_sum = math_ops.reduce_sum(outputs)
      self.assertTrue(outputs_sum.eval() == 1.)


if __name__ == '__main__':
  test.main()
