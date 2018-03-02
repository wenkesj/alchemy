# -*- coding: utf-8 -*-
import tensorflow as tf

from alchemy.layers import conv2d_rnn_encoder, conv2d_decoder
from alchemy.utils import product

from .zero_shot import ZeroShotImitationModel


class VisualImitationModel(ZeroShotImitationModel):

  def build_state_embedding_op(self,
                               filters=[1],
                               kernel_sizes=[1],
                               strides=[1],
                               activation=tf.nn.relu,
                               latent_hidden_sizes=[1],
                               latent_hidden_activation=tf.nn.relu,
                               rnn_hidden_sizes=[1],
                               rnn_cell_fn=tf.contrib.rnn.BasicLSTMCell):
    (outputs, states, initial_state_phs, zero_states,
     encoder, shapes, output_shape, hidden_sizes) = conv2d_rnn_encoder(
        self.state_ph,
        self.state_space.shape,
        filters,
        kernel_sizes,
        strides,
        activation,
        latent_hidden_sizes,
        latent_hidden_activation,
        rnn_hidden_sizes,
        rnn_cell_fn,
        'embedding',
        reuse=False)

    self.encoder = encoder
    self.encoder_shapes = shapes
    self.encoder_strides = strides
    self.encoder_output_shape = output_shape
    self.encoder_activation = activation
    self.latent_hidden_sizes = hidden_sizes
    self.latent_hidden_activation = latent_hidden_activation

    self.zero_state_fns.append(zero_states)
    self.internal_state_phs.append(initial_state_phs)
    self.internal_state.append(states)
    return outputs

  def build_goal_state_embedding_op(self,
                                    filters=[1],
                                    kernel_sizes=[1],
                                    strides=[1],
                                    activation=tf.nn.relu,
                                    latent_hidden_sizes=[1],
                                    latent_hidden_activation=tf.nn.relu,
                                    rnn_hidden_sizes=[1],
                                    rnn_cell_fn=tf.contrib.rnn.BasicLSTMCell):
    (outputs, states, initial_state_phs, zero_states, _, _, _, _) = conv2d_rnn_encoder(
        self.goal_state_ph,
        self.state_space.shape,
        filters,
        kernel_sizes,
        strides,
        activation,
        latent_hidden_sizes,
        latent_hidden_activation,
        rnn_hidden_sizes,
        rnn_cell_fn,
        'embedding',
        reuse=True)

    self.zero_state_fns.append(zero_states)
    self.internal_state_phs.append(initial_state_phs)
    self.internal_state.append(states)

    sequence_length = tf.shape(self.state_embedding_op)[1] - tf.shape(self.goal_state_ph)[1] + 1
    return tf.tile(outputs, [1, sequence_length] + [1] * len(outputs.get_shape()[2:]))

  def build_next_state_op(self, state_reconstruction_fn=None):
    with tf.variable_scope('next_state'):
      action_value = self.action_value_ph
      # if self.optimize_state and self.optimize_action_value:
      #   action_value = self.action_value_op
      inputs = tf.concat([self.state_embedding_op, action_value], axis=-1)
      inputs_shape = tf.shape(inputs)
      batch_size = inputs_shape[0]
      sequence_length = inputs_shape[1]

      x = inputs
      for hidden in self.latent_hidden_sizes[::-1]:
        x = tf.layers.dense(
            x, hidden,
            activation=self.latent_hidden_activation)
      x = tf.reshape(x, self.encoder_output_shape)

      x = conv2d_decoder(
          x,
          self.encoder,
          self.encoder_shapes,
          self.encoder_strides,
          scope='decoder',
          activation=self.encoder_activation)

      x = tf.reshape(x, [batch_size, sequence_length] + self.state_space.shape)

      if state_reconstruction_fn:
        self.state_reconstruction_summary_op = state_reconstruction_fn(
            x[-1, -1], name='state_reconstruction')
        self.summary_ops.append(self.state_reconstruction_summary_op)

        self.state_ground_truth_summary_op = state_reconstruction_fn(
            self.next_state_ph[-1, -1], name='state_truth')
        self.summary_ops.append(self.state_ground_truth_summary_op)
      return x

  def build_action_value_op(self,
                            hidden_sizes=[1],
                            activation=tf.nn.relu,
                            action_value_reconstruction_fn=None):
    with tf.variable_scope('action_value'):
      inputs = tf.concat([self.state_embedding_op, self.goal_state_embedding_op], axis=-1)
      x = inputs
      for hidden in hidden_sizes:
        x = tf.layers.dense(
            x, hidden,
            activation=activation)
      x = tf.layers.dense(x, product(self.action_space.shape))

      if action_value_reconstruction_fn:
        self.action_value_reconstruction_summary_op = action_value_reconstruction_fn(
            tf.nn.softmax(x[-1, -1]), name='action_value_reconstruction')
        self.summary_ops.append(self.action_value_reconstruction_summary_op)

        self.action_value_ground_truth_summary_op = action_value_reconstruction_fn(
            self.action_value_ph[-1, -1], name='action_value_truth')
        self.summary_ops.append(self.action_value_ground_truth_summary_op)
      return x

  def build_stop_criterion_op(self,
                              hidden_sizes=[1],
                              activation=tf.nn.relu):
    with tf.variable_scope('stop_criterion'):
      inputs = tf.concat([self.state_embedding_op, self.goal_state_embedding_op], axis=-1)
      x = inputs
      for hidden in hidden_sizes:
        x = tf.layers.dense(
            x, hidden,
            activation=activation)
      x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
      return x
