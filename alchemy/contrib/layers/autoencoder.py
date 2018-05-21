# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import core

from alchemy.contrib.rnn import stacked_rnn_impl


# TODO(wenkesj): this can easily be replaced with a better api -_-
def conv2d_encoder(inputs, filters, kernel_sizes, strides,
                   scope=None,
                   activation=None,
                   reuse=None):
  with variable_scope.variable_scope(scope, default_name="encoder", reuse=reuse) as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    if not context.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    encoder = []
    shapes = []

    for idx, n_outputs in enumerate(filters):
      n_input = inputs.get_shape().as_list()[3]
      shapes.append(inputs.get_shape().as_list())
      W = variable_scope.get_variable(
          'w_{}'.format(idx),
          [kernel_sizes[idx], kernel_sizes[idx], n_input, n_outputs],
          initializer=init_ops.variance_scaling_initializer())
      b = variable_scope.get_variable(
          'b_encoder_{}'.format(idx),
          [n_outputs], inputs.dtype,
          initializer=init_ops.zeros_initializer())
      encoder.append(W)
      outputs = math_ops.add(
          nn_ops.conv2d(
              inputs, W,
              strides=[1, strides[idx], strides[idx], 1],
              padding='SAME'), b)
      if activation:
        outputs = activation(outputs)

      inputs = outputs
    return inputs, encoder, shapes


def conv2d_decoder(inputs, encoder, shapes, strides,
                   scope=None,
                   activation=None,
                   weight_sharing=False,
                   reuse=None):
  with variable_scope.variable_scope(scope, default_name="decoder", reuse=reuse) as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    if not context.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    encoder.reverse()
    shapes.reverse()
    strides.reverse()
    for idx, shape in enumerate(shapes):
      encoder_W = encoder[idx]
      dtype = encoder_W.dtype
      W = encoder_W if weight_sharing else variable_scope.get_variable(
          'w_{}'.format(idx),
          encoder_W.get_shape().as_list(), dtype,
          initializer=init_ops.variance_scaling_initializer())
      b = variable_scope.get_variable(
          'b_decoder_{}'.format(idx),
          [W.get_shape().as_list()[2]], dtype,
          initializer=init_ops.zeros_initializer())
      outputs = math_ops.add(
          nn_ops.conv2d_transpose(
              inputs, W,
              array_ops.stack(
                  [array_ops.shape(inputs)[0], shape[1], shape[2], shape[3]]),
              strides=[1, strides[idx], strides[idx], 1],
              padding='SAME'), b)
      if activation:
        outputs = activation(outputs)
      inputs = outputs
    return inputs


# TODO(wenkesj): make this less crazy, replace it with better API
def conv2d_rnn_encoder(inputs,
                       input_shape,
                       filters,
                       kernel_sizes,
                       strides,
                       activation,
                       latent_hidden_sizes,
                       latent_hidden_activation,
                       rnn_hidden_sizes,
                       rnn_cell_fn,
                       scope=None,
                       reuse=None):
  with variable_scope.variable_scope(scope, default_name=scope, reuse=reuse):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    sequence_length = inputs_shape[1]
    stacked_inputs = gen_array_ops.reshape(
        inputs, [batch_size * sequence_length] + input_shape)

    x, encoder, shapes = conv2d_encoder(
        stacked_inputs, filters,
        kernel_sizes, strides,
        scope='encoder',
        activation=activation,
        reuse=reuse)
    output_shape = array_ops.shape(x)

    x = core.flatten(x)
    x_size = x.get_shape()[-1]
    x = gen_array_ops.reshape(
        x, [batch_size, sequence_length, x_size])

    for hidden in latent_hidden_sizes:
      x = core.dense(
          x, hidden,
          activation=latent_hidden_activation)

    outputs, states, initial_state_phs, zero_states = stacked_rnn_impl.stacked_rnn(
        x, rnn_hidden_sizes, rnn_cell_fn,
        scope='stacked_rnn',
        reuse=reuse)
    return ( # TODO(wenkesj): make this less crazy
        outputs, states, initial_state_phs, zero_states,
        encoder, shapes, output_shape, [x_size] + latent_hidden_sizes)
