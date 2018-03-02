# -*- coding: utf-8 -*-
import tensorflow as tf

from alchemy.utils import gaussian_from_logits, loc_log_scale, safe_tf_dtype

from .space import Space


class ContinuousSpace(Space):
  # TODO, this is very abstract and needs some future attention
  # similar to https://github.com/unixpickle/anyrl-py
  #
  def __init__(self, low, high, shape, dtype=tf.float32):
    self.low = low
    self.high = high
    self._shape = shape
    self._dtype = safe_tf_dtype(dtype)

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  def build_sample_op(self, logits):
    return gaussian_from_logits(logits, self.low, self.high, self.shape).sample()

  def build_mode_op(self, logits):
    locs, _ = loc_log_scale(logits, self.low, self.high, self.shape)
    return locs

  def build_loss_op(self, truth, logits):
    truth_dist = gaussian_from_logits(
        tf.cast(truth, logits.dtype),
        self.low, self.high, self.shape)
    logits_dist = gaussian_from_logits(
        logits,
        self.low, self.high, self.shape)
    return tf.distributions.kl_divergence(truth_dist, logits_dist)
