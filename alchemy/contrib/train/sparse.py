# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import adam
from tensorflow.python.training import optimizer


class SparseVariableOptimizer(optimizer.Optimizer):
  def __init__(self, opt,
               use_locking=False,
               name="SparseVariableOptimizer"):
    super(SparseVariableOptimizer, self).__init__(use_locking, name)
    self._opt = opt

  def get_initializer(self, var_list):
    slots = [self._opt.get_slot(var, name)
             for name in self._opt.get_slot_names()
             for var in var_list if var is not None]
    if isinstance(self._opt, adam.AdamOptimizer):
      slots.extend(list(self._opt._get_beta_accumulators()))
    return variables.variables_initializer(slots)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    masked_grads_and_vars = [(_mask_by_value(g, v), v)
                             for g, v in grads_and_vars]
    return self._opt.apply_gradients(masked_grads_and_vars, *args, **kwargs)


def _mask_by_value(inputs, masking_inputs):
  zero = constant_op.constant(0, dtype=masking_inputs.dtype)
  mask = math_ops.cast(gen_math_ops.not_equal(masking_inputs, zero), inputs.dtype)
  return math_ops.multiply(inputs, mask)
