# -*- coding: utf-8 -*-
from collections import deque

from .memory import Memory


class FIFOMemory(Memory):
  def __init__(self,
               state_dtype, state_shape,
               action_dtype, action_shape,
               action_value_dtype, action_value_shape,
               capacity):
    self.ram = deque([], capacity)
    self._state_dtype = state_dtype
    self._state_shape = state_shape
    self._action_dtype = action_dtype
    self._action_shape = action_shape
    self._action_value_dtype = action_value_dtype
    self._action_value_shape = action_value_shape

  @property
  def state_dtype(self):
    return self._state_dtype

  @property
  def state_shape(self):
    return self._state_shape

  @property
  def action_dtype(self):
    return self._action_dtype

  @property
  def action_shape(self):
    return self._action_shape

  @property
  def action_value_dtype(self):
    return self._action_value_dtype

  @property
  def action_value_shape(self):
    return self._action_value_shape

  def write(self, memory):
    self.ram.append(memory)

  def read(self):
    return self.ram.popleft()

  def clear(self):
    return self.ram.clear()

  def __len__(self):
    return len(self.ram)
