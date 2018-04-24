# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib import layers


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

_FastWeightsStateTuple = collections.namedtuple("FastWeightsStateTuple", ("c", "a"))


class FastWeightsStateTuple(_FastWeightsStateTuple):
  """Tuple used by FastWeights Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, A)`, in that order. Where `c` is the hidden state
  and `A` is the fast weights state.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, a) = self
    if c.dtype != a.dtype:
      raise TypeError("Inconsistent internal state: {} vs {}".format(
                      (str(c.dtype), str(a.dtype))))
    return c.dtype


# TODO(wenkesj): Add docstring
class FastWeightsRNNCell(rnn_cell_impl.LayerRNNCell):

  """
  Original Fast Weights
  https://arxiv.org/abs/1610.06258
  """

  def __init__(self,
               num_units,
               use_layer_norm=True,
               activation=nn_ops.relu,
               loop_steps=1,
               fast_learning_rate=.5,
               fast_decay_rate=0.95,
               layer_norm_scope="layer_norm",
               reuse=None,
               name=None):
    super(FastWeightsRNNCell, self).__init__(_reuse=reuse, name=name)
    self._num_units = num_units
    self._activation = activation
    self._use_layer_norm = use_layer_norm
    self._S = loop_steps
    self._eta = fast_learning_rate
    self._lambda = fast_decay_rate
    self._layer_norm_scope = layer_norm_scope

  @property
  def state_size(self):
    return FastWeightsStateTuple(
        self._num_units, tensor_shape.TensorShape(
            [self._num_units, self._num_units]))

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    self._kernel_w = self.add_variable(
        "{}_w".format(_WEIGHTS_VARIABLE_NAME),
        [self.output_size, self.output_size],
        dtype=self.dtype,
        initializer=init_ops.identity_initializer(gain=.05))
    self._kernel_c = self.add_variable(
        "{}_c".format(_WEIGHTS_VARIABLE_NAME),
        [inputs_shape[1], self.output_size],
        dtype=self.dtype)
    self._bias_c = self.add_variable(
        "{}_c".format(_BIAS_VARIABLE_NAME),
        [self.output_size],
        dtype=self.dtype)

  def call(self, inputs, state, scope=None):
    hidden_state, fast_weights = state

    batch_size = array_ops.shape(fast_weights)[0]
    add = math_ops.add
    multiply = math_ops.multiply

    slow = add(
        math_ops.matmul(hidden_state, self._kernel_w),
        nn_ops.bias_add(
            math_ops.matmul(inputs, self._kernel_c), self._bias_c))
    hidden_state = self._activation(slow)
    fast_weights = add(
        multiply(self._lambda, fast_weights),
        multiply(self._eta, special_math_ops.einsum(
            "ki,kj->ij", hidden_state, hidden_state)))
    hidden_state = array_ops.expand_dims(hidden_state, 1)
    h = tf.identity(hidden_state)
    for i in range(self._S):
      dot = math_ops.matmul(h, fast_weights)
      inner = add(slow, dot)
      h = self._activation(
          layers.layer_norm(inner, reuse=True)
          if self._use_layer_norm else inner)
    hidden_state = gen_array_ops.reshape(h, [batch_size, self._num_units])
    return hidden_state, FastWeightsStateTuple(hidden_state, fast_weights)


# TODO(wenkesj): Add docstring
class FastWeightsLSTMCell(rnn_cell_impl.LayerRNNCell):

  """
  Fast Weights + LSTM
  https://arxiv.org/pdf/1804.06511v1.pdf
  """

  def __init__(self,
               num_units,
               use_layer_norm=True,
               activation=nn_ops.relu,
               loop_steps=1,
               fast_learning_rate=.5,
               fast_decay_rate=0.95,
               forget_bias=1.,
               layer_norm_scope="layer_norm",
               reuse=None,
               name=None):
    super(FastWeightsLSTMCell, self).__init__(_reuse=reuse, name=name)
    self._num_units = num_units
    self._activation = activation or nn_ops.relu
    self._use_layer_norm = use_layer_norm
    self._S = loop_steps
    self._eta = fast_learning_rate
    self._lambda = fast_decay_rate
    self._forget_bias = forget_bias
    self._layer_norm_scope = layer_norm_scope

  @property
  def state_size(self):
    return FastWeightsStateTuple(
        rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units),
        tensor_shape.TensorShape([self._num_units, self._num_units]))

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    (c, h), fast_weights = state
    batch_size = array_ops.shape(fast_weights)[0]
    add = math_ops.add
    multiply = math_ops.multiply
    sigmoid = math_ops.sigmoid

    # Parameters of gates are concatenated into one multiply for efficiency.
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    if self._use_layer_norm:
      gate_inputs = layers.layer_norm(gate_inputs)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=1)

    fast_j = self._activation(j)
    fast_weights = add(
        multiply(self._lambda, fast_weights),
        multiply(self._eta, special_math_ops.einsum(
            "ki,kj->ij", fast_j, fast_j)))
    fast_weights_j = math_ops.matmul(
        gen_array_ops.reshape(fast_j, [batch_size, 1, -1]),
        fast_weights)
    fast_weights_j = gen_array_ops.reshape(fast_weights_j, [batch_size, self._num_units])
    fast_j = self._activation(add(fast_j, fast_weights_j))

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    new_c = add(multiply(c, sigmoid(add(f, self._forget_bias))),
                multiply(sigmoid(i), fast_j))

    if self._use_layer_norm:
      new_c = layers.layer_norm(new_c)

    new_h = multiply(self._activation(new_c), sigmoid(o))

    return new_h, FastWeightsStateTuple(
        rnn_cell_impl.LSTMStateTuple(new_c, new_h), fast_weights)
