# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty


class Memory(ABC):
  """Write and read a single example."""
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
