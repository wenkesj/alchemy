# -*- coding: utf-8 -*-
import tensorflow as tf

from .utils import ranged_axes


def loc_log_scale(logits, low, high, shape):
  # probably wrong,
  # TODO needs work, similar https://github.com/unixpickle/anyrl-py
  low_, high_ = tf.cast(low, logits.dtype), tf.cast(high, logits.dtype)
  means, variances = tf.nn.moments(logits, axes=ranged_axes(shape), keep_dims=True)
  log_stddevs = tf.log(tf.sqrt(variances))
  bias = (high_ + low_) / 2
  scale = (high_ - low_) / 2
  return means + bias, log_stddevs + tf.log(scale)


def gaussian_from_logits(logits, low, high, shape):
  # probably wrong,
  # TODO needs work, similar to https://github.com/unixpickle/anyrl-py
  locs, log_stddevs = loc_log_scale(logits, low, high, shape)
  return tf.distributions.Normal(loc=locs, scale=tf.exp(log_stddevs))
