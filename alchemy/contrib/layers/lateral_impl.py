# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.layers import base


class Lateral(base.Layer):
  """Lateral-connected layer class.
  This layer adds lateral connections.
  Arguments:
    allow_self_connections: Boolean `Tensor`. Scalar representing if the
      lateral connections should have self-connections, (e.g. kernel diagonal
      is non-zero). Default is `False`.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    kernel_trainable: Boolean, kernel should be added to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Properties:
    allow_self_connections: Boolean `Tensor`. Scalar representing if the
      lateral connections should have self-connections.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self,
               allow_self_connections=False,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_trainable=True,
               trainable=True,
               name=None,
               **kwargs):
    super(Lateral, self).__init__(trainable=trainable, name=name,
                                  activity_regularizer=activity_regularizer,
                                  **kwargs)
    self.allow_self_connections = ops.convert_to_tensor(
        allow_self_connections, dtype=dtypes.bool)
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.kernel_trainable = kernel_trainable
    self.input_spec = base.InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    input_dim = input_shape[-1].value
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_dim})
    self.kernel = self.add_variable('kernel',
                                    shape=[input_dim, input_dim],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=self.kernel_trainable)

    self.kernel = control_flow_ops.cond(
        self.allow_self_connections,
        true_fn=lambda: self.kernel,
        false_fn=lambda: gen_array_ops.matrix_set_diag(
            self.kernel, array_ops.zeros(input_dim)))

    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[input_dim,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    if len(shape) > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                             [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        outputs.set_shape(shape)
    else:
      outputs = math_ops.matmul(inputs, self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)

    # Add pre-activations
    outputs = math_ops.add(inputs, outputs)

    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)


def lateral(
    inputs,
    allow_self_connections=False,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    kernel_trainable=True,
    trainable=True,
    name=None,
    reuse=None):
  """Functional interface for the lateral-connected layer.
  This layer adds lateral connections.
  Arguments:
    inputs: Tensor input.
    allow_self_connections: Boolean `Tensor`. Scalar representing if the
      lateral connections should have self-connections, (e.g. kernel diagonal
      is non-zero). Default is `False`.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    kernel_trainable: Boolean, kernel should be added to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Returns:
    Output tensor the same shape as `inputs` except the last dimension is of
    size `units`.
  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Lateral(allow_self_connections=allow_self_connections,
                  activation=activation,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  kernel_constraint=kernel_constraint,
                  bias_constraint=bias_constraint,
                  trainable=trainable,
                  name=name,
                  dtype=inputs.dtype.base_dtype,
                  _scope=name,
                  _reuse=reuse)
  return layer.apply(inputs)
