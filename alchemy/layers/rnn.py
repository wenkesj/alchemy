# -*- coding: utf-8 -*-
import tensorflow as tf

# tf.placeholder(dtype, [None, size], name='state_{}'.format(idx))
def _create_initial_state(state_size, cell_type, dtype):
  if isinstance(state_size, tuple):
    states = []
    for size in state_size:
      states.append(_create_initial_state(size, cell_type, dtype))
    if state_size.__class__ == tuple:
      states = tuple(states)
    else:
      states = state_size.__class__(*states)
    return states
  if isinstance(state_size, tf.TensorShape):
    return tf.placeholder(dtype, [None,] + state_size.as_list())
  state_in = tf.placeholder(dtype, [None, state_size])
  return state_in


def maybe_dropout_cell(dropout, hidden_size, cell_fn):
  cell = cell_fn(hidden_size)
  cell_type = cell.__class__
  if dropout:
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.-dropout)
  return cell, cell_type


def stacked_rnn(inputs, hidden_sizes, cell_fn, scope, dropouts=None, dtype=tf.float32, reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    if dropouts is None:
      dropouts = [None] * len(hidden_sizes)

    layers, cell_types = [], []
    fixed_hidden_sizes = hidden_sizes + [hidden_sizes[-1]]
    for idx, (dropout, hidden_size) in enumerate(zip(dropouts, fixed_hidden_sizes[:-1])):
      cell, cell_type = maybe_dropout_cell(
          dropout, hidden_size, cell_fn)
      if hidden_size != fixed_hidden_sizes[idx + 1]:
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, fixed_hidden_sizes[idx + 1])
      layers.append(cell)
      cell_types.append(cell_type)

    initial_states = tuple(
        [_create_initial_state(cell.state_size, cell_type, dtype)
         for cell, cell_type in zip(layers, cell_types)])
    layers = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(
        layers, inputs,
        initial_state=initial_states,
        dtype=dtype,
        time_major=False)
    return outputs, states, initial_states, layers.zero_state
