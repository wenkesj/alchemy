# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from alchemy.contrib.rnn import stacked_rnn_impl


# TODO(wenkesj): make this a set of flavors, like this one is vanilla
def conv2d_encoder(inputs,
                   filters,
                   kernel_sizes,
                   strides,
                   scope,
                   activation=None,
                   reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    encoder = []
    shapes = []

    for idx, n_outputs in enumerate(filters):
      n_input = inputs.get_shape().as_list()[3]
      shapes.append(inputs.get_shape().as_list())
      W = tf.get_variable(
          'w_{}'.format(idx),
          [kernel_sizes[idx], kernel_sizes[idx], n_input, n_outputs],
          initializer=tf.initializers.variance_scaling())
      b = tf.get_variable(
          'b_encoder_{}'.format(idx),
          [n_outputs], tf.float32,
          initializer=tf.initializers.zeros())
      encoder.append(W)
      outputs = tf.add(
          tf.nn.conv2d(
              inputs, W,
              strides=[1, strides[idx], strides[idx], 1],
              padding='SAME'), b)
      if activation:
        outputs = activation(outputs)

      inputs = outputs
    return inputs, encoder, shapes


def conv2d_decoder(inputs,
                   encoder,
                   shapes,
                   strides,
                   scope,
                   activation=None,
                   weight_sharing=False,
                   reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    encoder.reverse()
    shapes.reverse()
    strides.reverse()
    for idx, shape in enumerate(shapes):
      encoder_W = encoder[idx]
      W = encoder_W if weight_sharing else tf.get_variable(
          'w_{}'.format(idx),
          encoder_W.get_shape().as_list(), tf.float32,
          initializer=tf.initializers.variance_scaling())
      b = tf.get_variable(
          'b_decoder_{}'.format(idx),
          [W.get_shape().as_list()[2]], tf.float32,
          initializer=tf.initializers.zeros())
      outputs = tf.add(
          tf.nn.conv2d_transpose(
              inputs, W,
              tf.stack(
                  [tf.shape(inputs)[0], shape[1], shape[2], shape[3]]),
              strides=[1, strides[idx], strides[idx], 1],
              padding='SAME'), b)
      if activation:
        outputs = activation(outputs)
      inputs = outputs
    return inputs


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
                       scope,
                       reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    sequence_length = inputs_shape[1]
    stacked_inputs = tf.reshape(
        inputs, [batch_size * sequence_length] + input_shape)

    x, encoder, shapes = conv2d_encoder(
        stacked_inputs, filters,
        kernel_sizes, strides,
        scope='encoder',
        activation=activation,
        reuse=reuse)
    output_shape = tf.shape(x)

    x = tf.layers.flatten(x)
    x_size = x.get_shape()[-1]
    x = tf.reshape(
        x, [batch_size, sequence_length, x_size])

    for hidden in latent_hidden_sizes:
      x = tf.layers.dense(
          x, hidden,
          activation=latent_hidden_activation)

    outputs, states, initial_state_phs, zero_states = stacked_rnn_impl.stacked_rnn(
        x, rnn_hidden_sizes, rnn_cell_fn,
        scope='stacked_rnn',
        reuse=reuse)
    return ( # TODO(wenkesj): make this less crazy
        outputs, states, initial_state_phs, zero_states,
        encoder, shapes, output_shape, [x_size] + latent_hidden_sizes)
