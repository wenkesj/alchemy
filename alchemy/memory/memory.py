# -*- coding: utf-8 -*-
from __future__ import absolute_import

from abc import ABC, abstractmethod, abstractproperty


# TODO(wenkesj): Add docstring
# TODO(wenkesj): Design a better process to make this super simplified.
# i.e. factories for memory declaration, like `collections`
class Memory(ABC):
  @abstractproperty
  def state_dtype(self):
    pass

  @abstractproperty
  def state_shape(self):
    pass

  @abstractproperty
  def action_dtype(self):
    pass

  @abstractproperty
  def action_shape(self):
    pass

  @abstractproperty
  def action_value_dtype(self):
    pass

  @abstractproperty
  def action_value_shape(self):
    pass

  @abstractmethod
  def write(self, memory):
    pass

  @abstractmethod
  def read(self):
    pass

  @abstractmethod
  def clear(self):
    pass

  @abstractmethod
  def __len__(self):
    pass
