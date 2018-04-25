# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from sonnet.python.custom_getters import bayes_by_backprop

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.client import session
from tensorflow.python.layers import core

from alchemy.contrib.train import bbb


def mlp(inputs_ph, hidden_layers):
  inputs = inputs_ph / 127
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden, units=hidden_size, use_bias=False, activation=nn_ops.relu)
  logits = core.dense(
      hidden, units=10, use_bias=False)
  return logits


class PruneBBBTest(test.TestCase):

  def test_prune_by_bbb_from_scratch_is_correct(self):
    hidden_layers = [256, 128]

    sampling_mode_ph = array_ops.placeholder(dtypes.string, [])
    get_bbb_variable_fn = bayes_by_backprop.bayes_by_backprop_getter(
        posterior_builder=bayes_by_backprop.diagonal_gaussian_posterior_builder,
        prior_builder=bayes_by_backprop.adaptive_gaussian_prior_builder,
        kl_builder=bayes_by_backprop.stochastic_kl_builder,
        sampling_mode_tensor=sampling_mode_ph)

    # create the bayes network
    bbb_scope = 'net'
    inputs_ph = array_ops.placeholder(dtypes.float32, [None, 784])
    with variable_scope.variable_scope(bbb_scope, custom_getter=get_bbb_variable_fn) as vs:
      logits_bbb = mlp(inputs_ph, hidden_layers)

    # create the pruning op
    metadata = bayes_by_backprop.get_variable_metadata()
    total_variables = sum([np.prod(meta.raw_variable_shape) for meta in metadata])
    percentage_ph = array_ops.placeholder(dtypes.float32, [])
    pruned_vars_op = bbb.prune_by_bbb(metadata, percentage_ph)

    # create the template network
    template_scope = 'template'
    with variable_scope.variable_scope(template_scope) as vs:
      logits = mlp(inputs_ph, hidden_layers)
      # retreve the variables so we can prune them in-place
      template_variables = variables.trainable_variables(scope=template_scope)

    # find the variables from 'net' that correspond to 'template'
    assign_pruned_vars_op = bbb.assign_pruned_by_bbb_to_template(
        metadata, pruned_vars_op, template_variables,
        from_scope=bbb_scope, to_scope=template_scope)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      for percentage in np.arange(0., 1., .05):
        # ordinary pruning without the test-case looks like this:
        # >>> sess.run(assign_pruned_vars_op, feed_dict={percentage_ph: percentage})

        test_variables, _ = sess.run(
            (pruned_vars_op, assign_pruned_vars_op),
            feed_dict={
              percentage_ph: percentage,
              sampling_mode_ph: bayes_by_backprop.EstimatorModes.mean,
            })

        nonzero = 0
        for test_variable, template_variable in zip(test_variables, template_variables):
          nonzero += np.count_nonzero(test_variable)
          self.assertAllClose(test_variable, sess.run(template_variable), atol=1e-8)
        self.assertAlmostEqual(int(nonzero), int(total_variables * (1 - percentage)), delta=1)

  def test_prune_by_bbb_from_scratch_to_sparse_ops(self):
    hidden_layers = [256, 128]

    sampling_mode_ph = array_ops.placeholder(dtypes.string, [])
    get_bbb_variable_fn = bayes_by_backprop.bayes_by_backprop_getter(
        posterior_builder=bayes_by_backprop.diagonal_gaussian_posterior_builder,
        prior_builder=bayes_by_backprop.adaptive_gaussian_prior_builder,
        kl_builder=bayes_by_backprop.stochastic_kl_builder,
        sampling_mode_tensor=sampling_mode_ph)

    # create the bayes network
    bbb_scope = 'net'
    inputs_ph = array_ops.placeholder(dtypes.float32, [None, 784])
    with variable_scope.variable_scope(bbb_scope, custom_getter=get_bbb_variable_fn) as vs:
      logits_bbb = mlp(inputs_ph, hidden_layers)

    # create the optimal pruning op
    metadata = bayes_by_backprop.get_variable_metadata()
    total_variables = sum([np.prod(meta.raw_variable_shape) for meta in metadata])
    percentage_ph = array_ops.placeholder(dtypes.float32, [])
    pruned_vars_op = bbb.prune_by_bbb(metadata, percentage_ph)

    # create the template network
    template_scope = 'template'
    with variable_scope.variable_scope(template_scope) as vs:
      sparse_logits = mlp(inputs_ph, hidden_layers)

      # retreve the variables so we can prune them in-place
      template_variables = variables.trainable_variables(scope=template_scope)

    # find the variables from 'net' that correspond to 'template'
    assign_pruned_vars_op = bbb.assign_pruned_by_bbb_to_template(
        metadata, pruned_vars_op, template_variables,
        from_scope=bbb_scope, to_scope=template_scope)

    with self.test_session() as sess:
      test_inputs_ones = np.ones([1, 784])
      sess.run(variables.global_variables_initializer())
      for percentage in np.arange(0., 1.01, .01):
        sess.run(
            assign_pruned_vars_op, feed_dict={
              percentage_ph: percentage,
              sampling_mode_ph: bayes_by_backprop.EstimatorModes.mean,
            })
        sess.run(sparse_logits, feed_dict={inputs_ph: test_inputs_ones})


if __name__ == '__main__':
  test.main()
