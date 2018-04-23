# -*- coding: utf-8 -*-
from __future__ import absolute_import

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf

from alchemy.utils import array_utils
from alchemy.utils import sequence_utils
from alchemy.utils import type_utils
from alchemy.utils import shortcuts


class ZeroShotImitationModel(ABC):
  def __init__(self, state_space, action_space,
               optimize_state, optimize_action_value):
    self.global_step = tf.train.get_or_create_global_step()

    self.state_space = state_space
    self.action_space = action_space

    self.optimize_state = optimize_state
    self.optimize_action_value = optimize_action_value

    self.sequence_length_ph = tf.placeholder(
        tf.int32, [None],
        name='sequence_length')
    self.state_ph = tf.placeholder(
        self.state_space.dtype,
        [None, None] + self.state_space.shape,
        name='state')
    self.next_state_ph = tf.placeholder(
        self.state_space.dtype,
        [None, None] + self.state_space.shape,
        name='next_state')
    self.action_value_ph = tf.placeholder(
        tf.float32,
        [None, None] + self.action_space.shape,
        name='action_value')
    self.goal_state_ph = tf.placeholder(
        self.state_space.dtype,
        [None, None] + self.state_space.shape,
        name='goal_state')

    self.zero_state_fns = []
    self.internal_state_phs = []
    self.internal_state = []
    self.summary_ops = []

  @abstractmethod
  def build_state_embedding_op(self, **kwargs):
    pass

  @abstractmethod
  def build_goal_state_embedding_op(self, **kwargs):
    pass

  @abstractmethod
  def build_action_value_op(self, **kwargs):
    pass

  @abstractmethod
  def build_next_state_op(self, **kwargs):
    pass

  @abstractmethod
  def build_stop_criterion_op(self, **kwargs):
    pass

  def build_action_op(self, **kwargs):
    return self.action_space.build_mode_op(self.action_value_op)

  def build_next_state_loss_op(self, **kwargs):
    return .5 * shortcuts.ssd(self.next_state_op, self.next_state_ph)

  def build_action_value_loss_op(self, **kwargs):
    return self.action_space.build_loss_op(
        self.action_value_ph, self.action_value_op)

  def build_stop_criterion_loss_op(self, **kwargs):
    distance = tf.sqrt(shortcuts.ssd(self.state_ph, self.goal_state_ph))
    max_state = self.state_space.high * tf.ones_like(self.state_ph)
    min_state = self.state_space.low * tf.ones_like(self.state_ph)
    max_distance = tf.sqrt(shortcuts.ssd(max_state, min_state))
    norm = distance / max_distance
    return tf.abs(tf.squeeze(self.stop_criterion_op, -1) - norm)

  def build_optimize_op(self, **kwargs):
    self.total_loss = 0.
    sequence_length_mask, sequence_length_total = sequence_utils.mask_sequence(
        self.sequence_length_ph, kwargs['max_sequence_length'])

    if self.optimize_state:
      self.next_state_loss_op = kwargs['l2coeff'] * tf.reduce_sum(
          sequence_length_mask * self.build_next_state_loss_op(), axis=-1)
      self.next_state_loss_op /= sequence_length_total
      self.total_loss += self.next_state_loss_op
      self.next_state_loss_summary_op = tf.summary.scalar(
          'next_state_loss', tf.reduce_mean(self.next_state_loss_op))
      self.summary_ops.append(self.next_state_loss_summary_op)

    if self.optimize_action_value:
      self.action_value_loss_op = tf.reduce_sum(
          sequence_length_mask * self.build_action_value_loss_op(), axis=-1)
      self.action_value_loss_op /= sequence_length_total
      self.total_loss += self.action_value_loss_op
      self.action_value_loss_summary_op = tf.summary.scalar(
          'action_value_loss', tf.reduce_mean(self.action_value_loss_op))
      self.summary_ops.append(self.action_value_loss_summary_op)

    if self.optimize_state or self.optimize_action_value:
      self.stop_criterion_loss_op = tf.reduce_sum(
          sequence_length_mask * self.build_stop_criterion_loss_op(), axis=-1)
      self.stop_criterion_loss_op /= sequence_length_total
      self.total_loss += self.stop_criterion_loss_op
      self.stop_criterion_loss_summary_op = tf.summary.scalar(
          'stop_criterion_loss', tf.reduce_mean(self.stop_criterion_loss_op))
      self.summary_ops.append(self.stop_criterion_loss_summary_op)

    self.total_loss = tf.reduce_mean(self.total_loss)
    self.total_loss_summary_op = tf.summary.scalar('total_loss', self.total_loss)
    self.summary_ops.append(self.total_loss_summary_op)

    return tf.train.AdamOptimizer(kwargs['learning_rate']).minimize(
        self.total_loss, global_step=self.global_step)

  def zero_state_op(self, batch_size, dtype=tf.float32):
    states = [fn(batch_size, type_utils.safe_tf_dtype(dtype)) for fn in self.zero_state_fns]
    return states

  def build(self,
            state_embedding_kwargs={},
            goal_state_embedding_kwargs={},
            action_value_kwargs={},
            next_state_kwargs={},
            action_kwargs={},
            stop_criterion_kwargs={},
            optimize_kwargs={}):
    self.state_embedding_op = self.build_state_embedding_op(
        **state_embedding_kwargs)
    self.goal_state_embedding_op = self.build_goal_state_embedding_op(
        **goal_state_embedding_kwargs)
    self.action_value_op = self.build_action_value_op(**action_value_kwargs)
    if self.optimize_state and not self.optimize_action_value:
      self.action_value_op = tf.stop_gradient(self.action_value_op)
    self.next_state_op = self.build_next_state_op(**next_state_kwargs)
    if self.optimize_action_value and not self.optimize_state:
      self.next_state_op = tf.stop_gradient(self.next_state_op)
    self.action_op = self.build_action_op(**action_kwargs)
    self.stop_criterion_op = self.build_stop_criterion_op(**stop_criterion_kwargs)
    if self.optimize_state or self.optimize_action_value:
      self.optimize_op = self.build_optimize_op(**optimize_kwargs)
    self.summary_op = tf.summary.merge(self.summary_ops)
    return self

  def optimize_feed_dict(self, state, next_state, goal_state,
                         action_value, sequence_length,
                         initial_state=[], extra_feed_dict={}):
    internal_state = {k: v for k, v in zip(self.internal_state_phs, initial_state)}

    return {
      self.state_ph: state,
      self.next_state_ph: next_state,
      self.goal_state_ph: goal_state,
      self.sequence_length_ph: sequence_length,
      self.action_value_ph: action_value,
      **internal_state,
      **extra_feed_dict,
    }

  def next_state_feed_dict(self, state, action_value, initial_state=[], extra_feed_dict={}):
    assert array_utils.all_equal(state.shape, self.state_space.shape)
    assert array_utils.all_equal(action_value.shape, self.action_space.shape)

    internal_state = {k: v for k, v in zip(self.internal_state_phs, initial_state)}
    return {
      self.state_ph: array_utils.nd_expand_dims(state, 2),
      self.action_value_ph: array_utils.nd_expand_dims(action_value, 2),
      **internal_state,
      **extra_feed_dict,
    }

  def action_value_feed_dict(self, state, goal_state, initial_state=[], extra_feed_dict={}):
    assert array_utils.all_equal(state.shape, self.state_space.shape)
    assert array_utils.all_equal(goal_state.shape, self.state_space.shape)

    internal_state = {k: v for k, v in zip(self.internal_state_phs, initial_state)}
    return {
      self.state_ph: array_utils.nd_expand_dims(state, 2),
      self.goal_state_ph: array_utils.nd_expand_dims(goal_state, 2),
      **internal_state,
      **extra_feed_dict,
    }
