# -*- coding: utf-8 -*-
from __future__ import absolute_import

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope

from alchemy.utils import distribution_utils
from alchemy.utils import array_utils


def assign_pruned_by_bbb_to_template(metadata, pruned_variables, template_variables,
                                     from_scope=None, to_scope=None):
  """Assigns `pruned_variables` to `template_variables` by `bbb` `metadata`.

  Arguments:
    metadata: `list` of `bbb._VariableMetadata`, return from
        `bbb.get_variable_metadata()`.
    pruned_variables: `list` of `tf.SparseTensor`, returned from `prune`.
    template_variables: `list` of `tf.Variable` corresponding to `metadata`
        variables.
    from_scope: `str` representing the `pruned_variables` scope.
    to_scope: `str` representing the `template_variables` scope.
  """
  from_scope = from_scope + '/'
  to_scope = to_scope + '/'
  lookup_table = {variable.name.split(to_scope)[-1]: variable for variable in template_variables}
  assign_pruned_variables_ops = []
  for meta, pruned in zip(metadata, pruned_variables):
    scope = pruned.name.split(from_scope)[-1]
    if scope in lookup_table:
      assign_pruned_variables_ops.append(
        lookup_table[scope].assign(pruned))
  return control_flow_ops.group(*assign_pruned_variables_ops)


def prune_by_bbb(variable_metadata, percentage):
  """Prune a percentage of variables based on their signal to noise ratios.

  Arguments:
    variable_metadata: `list` of `bbb._VariableMetadata`, suggest using
        `bbb.get_variable_metadata()`.
    percentage: a `tf.Tensor` that is scalar representing what percentage
        of variables to prune.
  """
  if not variable_metadata:
    return []

  signal_to_noise_ratios = []
  variable_estimates = []
  variable_info = []

  # get signal to noise and mean posterior
  for meta in variable_metadata:
    posterior_dist = meta.posterior
    signal_to_noise_ratios.append(
        array_utils.flatten(
            distribution_utils.signal_to_noise_ratio(posterior_dist)))
    variable_estimates.append(array_utils.flatten(meta.posterior_estimate))
    variable_info.append((meta.raw_variable_name, meta.raw_variable_shape))

  # flatten variables
  flat_variable_estimates = array_ops.concat(variable_estimates, 0)
  flat_signal_to_noise_ratios = array_ops.concat(signal_to_noise_ratios, 0)
  flat_variable_size = flat_variable_estimates.get_shape().as_list()[-1]
  flat_drop_size = math_ops.cast(flat_variable_size * percentage, dtypes.int32)

  # sort by signal to noise ratio
  _, indices = nn_ops.top_k(
      flat_signal_to_noise_ratios,
      k=flat_variable_size,
      sorted=True)
  zero_indices = array_ops.expand_dims(indices[:flat_drop_size], -1)
  mask = math_ops.cast(
      sparse_ops.sparse_to_dense(
          zero_indices, [flat_variable_size],
          sparse_values=0, default_value=1,
          validate_indices=False),
      flat_variable_estimates.dtype)
  flat_variable_estimates *= mask

  # unflatten variables
  start = 0
  dsts = []
  for name, shape in variable_info:
    end = array_utils.product(shape)
    dst = gen_array_ops.reshape(
        flat_variable_estimates[start:start+end], shape, name=name)
    dsts.append(dst)
    start += end
  return dsts


def assign_template_to_prune_by_bbb(template_variables, metadata,
                                    from_scope=None, to_scope=None):
  """Assign the mean of a posterior `tf.distributions.Normal` from a template scope.

  Arguments:
    metadata: `list` of `bbb._VariableMetadata`, return from
        `bbb.get_variable_metadata()`.
    template_variables: `list` of `tf.Variable` corresponding to `metadata`
        variables.
    from_scope: `str` representing the `template_variables` scope.
    to_scope: `str` representing the `pruned_variables` scope.
  """
  lookup_table = {}
  for m in metadata:
    with variable_scope.variable_scope('', reuse=True):
      var = variable_scope.get_variable(
          '{}/posterior_loc'.format(m.raw_variable_name),
          shape=m.raw_variable_shape)
      lookup_table[m.raw_variable_name.split(to_scope)[-1] + ':0'] = var
  assign_ops = []
  for t in template_variables:
    name = t.name.split(from_scope)[-1]
    if name in lookup_table:
      assign_ops.append(lookup_table[name].assign(t))
  return control_flow_ops.group(*assign_ops)
