# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import base
from tensorflow.python.layers import utils


class WTA(base.Layer):
  """Applies WTA to the input.
  Dropout consists in keeping top-k activations and zero-ing out the rest.
  Arguments:
    k: The top-k, between 0 and `input_shape[-1]`.
    name: The name of the layer (string).
  """

  def __init__(self, k,
               name=None,
               **kwargs):
    super(WTA, self).__init__(name=name, **kwargs)
    self.k = ops.convert_to_tensor(k, dtype=dtypes.int32)

  def call(self, inputs, training=False):
    input_dim = inputs.get_shape()[-1].value
    _, indices = nn_ops.top_k(inputs, self.k, sorted=False)
    mask = array_ops.one_hot(indices, input_dim, axis=-1)
    mask = math_ops.reduce_sum(mask, axis=-2)
    return utils.smart_cond(
        training,
        lambda: mask * inputs,
        lambda: array_ops.identity(inputs))

  def compute_output_shape(self, input_shape):
    return input_shape


def wta(inputs, k, name=None, **kwargs):
  """Applies WTA to the input.
  Dropout consists in keeping top-k activations and zero-ing out the rest.
  Arguments:
    inputs: Tensor input.
    k: The top-k, between 0 and `input_shape[-1]`.
    name: The name of the layer (string).
  """
  layer = WTA(k, name=name, **kwargs)
  return layer.apply(inputs)


class DWTA(base.Layer):
  """Applies d-WTA to the input.
  Dropout consists in keeping top-k activations and zero-ing out the rest.
  Arguments:
    k: The top-k, between 0 and `input_shape[-1]`.
    name: The name of the layer (string).
  """

  def __init__(self,
               name=None,
               **kwargs):
    super(DWTA, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=False):
    input_dim = inputs.get_shape()[-1].value
    k = random_ops.random_uniform(
        [1], maxval=input_dim,
        dtype=dtypes.int32)[0]
    _, indices = nn_ops.top_k(inputs, k, sorted=False)
    mask = array_ops.one_hot(indices, input_dim, axis=-1)
    mask = math_ops.reduce_sum(mask, axis=-2)
    return utils.smart_cond(
        training,
        lambda: mask * inputs,
        lambda: array_ops.identity(inputs))

  def compute_output_shape(self, input_shape):
    return input_shape


def d_wta(inputs, name=None, **kwargs):
  """Applies d-WTA to the input.
  Dropout consists in keeping top-k activations and zero-ing out the rest.
  Arguments:
    inputs: Tensor input.
    k: The top-k, between 0 and `input_shape[-1]`.
    name: The name of the layer (string).
  """
  layer = DWTA(name=name, **kwargs)
  return layer.apply(inputs)
