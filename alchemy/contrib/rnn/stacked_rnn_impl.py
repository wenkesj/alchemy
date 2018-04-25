# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn as rnn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import rnn


# TODO(wenkesj): probably make this comform better with the framework.nest api.
#   i.e. right now this is recursively creating placeholders for all (nested)
#   states it finds and rewraps them with the class that created it. See {.1}
def create_initial_state_placeholder(state_size, dtype=dtypes.float32):
  if isinstance(state_size, tuple):
    states = []
    for size in state_size:
      states.append(create_initial_state_placeholder(size, dtype))
    if state_size.__class__ == tuple: # <- {.1}
      states = tuple(states) # <- {.1}
    else: # <- {.1}
      states = state_size.__class__(*states) # <- {.1}
    return states
  if isinstance(state_size, tensor_shape.TensorShape):
    return array_ops.placeholder(dtype, [None,] + state_size.as_list())
  state_in = array_ops.placeholder(dtype, [None, state_size])
  return state_in


# TODO(wenkesj): make this isomorphic and comforming to the tf API, like dynamic_rnn...
def stacked_rnn(inputs, hidden_sizes, cell_fn,
                scope=None,
                dtype=dtypes.float32,
                reuse=False):
  with variable_scope.variable_scope(scope or "stacked_rnn", reuse=reuse) as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if not context.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    layers = []
    fixed_hidden_sizes = hidden_sizes + [hidden_sizes[-1]]
    for idx, hidden_size in enumerate(fixed_hidden_sizes[:-1]):
      cell = cell_fn(hidden_size)
      if hidden_size != fixed_hidden_sizes[idx + 1]:
        cell = rnn.OutputProjectionWrapper(cell, fixed_hidden_sizes[idx + 1])
      layers.append(cell)
    initial_states = tuple(
        [create_initial_state_placeholder(
            cell.state_size, dtype) for cell in layers])
    layers = rnn.MultiRNNCell(layers)
    outputs, states = rnn_ops.dynamic_rnn(
        layers, inputs,
        initial_state=initial_states,
        dtype=dtype,
        time_major=False)
    return outputs, states, initial_states, layers.zero_state
