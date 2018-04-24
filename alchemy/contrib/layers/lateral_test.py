# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

from alchemy.contrib.layers import lateral_impl


class LateralTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_call_tensordot(self):
    layer = lateral_impl.Lateral(activation=nn_ops.relu, name='my_lateral')
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    outputs = layer(inputs)
    self.assertListEqual([5, 4, 3], outputs.get_shape().as_list())


if __name__ == '__main__':
  test.main()
