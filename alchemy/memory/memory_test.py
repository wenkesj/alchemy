# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import deque
import numpy as np
import unittest

from alchemy.memory import memory
from alchemy.memory import trajectory


hello_world = 'hello world'
vocab = list(set(hello_world))
e = vocab.index('e')
l = vocab.index('l')
o = vocab.index('o')
sp = vocab.index(' ')
w = vocab.index('w')
r = vocab.index('r')
d = vocab.index('d')
oh = np.eye(len(vocab))

class SimpleMemory(memory.Memory):
  def __init__(self):
    self.ram = deque([])

  @property
  def state_dtype(self):
    return str

  @property
  def state_shape(self):
    return []

  @property
  def action_dtype(self):
    return int

  @property
  def action_shape(self):
    return []

  @property
  def action_value_dtype(self):
    return float

  @property
  def action_value_shape(self):
    return [len(vocab)]

  def write(self, mem):
    self.ram.append(mem)

  def read(self):
    return self.ram.popleft()

  def clear(self):
    return self.ram.clear()

  def __len__(self):
    return len(self.ram)


class MemoryTest(unittest.TestCase):

  def test_simple_memory(self):
    ram = SimpleMemory()

    ram.write(
        trajectory.Trajectory(
            [trajectory.Transition('h', e, oh[e], 0, False, []),
             trajectory.Transition('e', l, oh[l], 0, False, []),
             trajectory.Transition('l', l, oh[l], 0, False, []),
             trajectory.Transition('l', o, oh[o], 0, False, []),
             trajectory.Transition('o', sp, oh[sp], 0, False, []),
             trajectory.Transition(' ', w, oh[w], 0, False, []),
             trajectory.Transition('w', o, oh[o], 0, False, []),
             trajectory.Transition('o', r, oh[r], 0, False, []),
             trajectory.Transition('r', l, oh[l], 0, False, []),
             trajectory.Transition('l', d, oh[d], 0, False, []),
             trajectory.Transition('d', sp, oh[sp], 1, True, [])]))

    self.assertTrue(len(ram) == 1)
    ram.clear()
    self.assertTrue(len(ram) == 0)


if __name__ == '__main__':
  unittest.main()
