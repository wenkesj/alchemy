# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops

from alchemy.contrib.rl import core_ops


def discount_py(rewards_batch, gamma=0.99):
  discounts_batch = []
  for rewards in rewards_batch:
    discounts = []
    R = 0
    for r in rewards[::-1]:
      R = r + gamma * R
      discounts.insert(0, R)
    discounts_batch.append(discounts)
  return discounts_batch


class CoreOpsTest(test.TestCase):

  def test_discount(self):
    t = self.evaluate

    def test_discount_(reward, gamma, expected):
      if len(np.asarray(reward).shape) == 1:
        length = len(reward)
        reward = [reward]
      else:
        length = len(reward[0])
      self.assertAllClose(
          np.squeeze(discount_py(reward, gamma=gamma)), expected)
      self.assertAllClose(
          t(array_ops.squeeze(core_ops.discount(reward, length, discount=gamma))), expected)

    # test singles
    test_discount_(
        [0, 0, 1, 0, 0, 1, 0], .9,
        [1.40049, 1.5561, 1.729, .81, .9, 1., 0.])
    test_discount_(
        [0, 0, 1, -2, 3, -4, 0], .5,
        [0.0625, .125, .25, -1.5, 1., -4.0, 0.])
    test_discount_(
        [0, 0, 1, 2, 3, 4, 0], .0,
        [0, 0, 1, 2, 3, 4, 0])

    # test mini-batches
    test_discount_(
        [[0, 0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0]], .9,
        [[1.40049, 1.5561, 1.729, .81, .9, 1., 0.],
         [1.40049, 1.5561, 1.729, .81, .9, 1., 0.]])


if __name__ == '__main__':
  test.main()
