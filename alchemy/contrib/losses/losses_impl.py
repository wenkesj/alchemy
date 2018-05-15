# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def huber_loss(x, delta=1.0):
    return array_ops.where(
        math_ops.abs(x) < delta,
        math_ops.square(x) * 0.5,
        delta * (math_ops.abs(x) - 0.5 * delta))
