# -*- coding: utf-8 -*-
import tensorflow as tf

from alchemy.utils import safe_tf_dtype

from .space import Space


class CategoricalSpace(Space):

  def __init__(self, shape, is_one_hot=True, dtype=tf.float32):
    self._shape = shape
    self._is_one_hot = is_one_hot
    self._dtype = safe_tf_dtype(dtype)

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  def build_sample_op(self, logits):
    return self.build_mode_op(logits)

  def build_mode_op(self, logits):
    value = tf.distributions.Categorical(logits=logits).sample()
    if self._is_one_hot:
      return tf.one_hot(value, self.shape[-1])
    return value

  def build_loss_op(self, truth, logits):
    truth_dist = tf.cast(truth, logits.dtype)
    logits_dist = tf.nn.softmax(logits)
    total_loss = -tf.reduce_sum(truth_dist * tf.log(logits_dist), axis=-1)
    return total_loss
