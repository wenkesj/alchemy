# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty


class Space(ABC):

  @abstractproperty
  def shape(self):
    pass

  @abstractproperty
  def dtype(self):
    pass

  @abstractmethod
  def build_sample_op(self, logits):
    pass

  @abstractmethod
  def build_mode_op(self, logits):
    pass

  @abstractmethod
  def build_loss_op(self, truth, logits):
    pass
