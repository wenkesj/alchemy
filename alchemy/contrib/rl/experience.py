# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections


class Experience(collections.namedtuple('Experience', ['state', 'next_state',
                                                       'action', 'action_value',
                                                       'reward', 'terminal'])):
  """
  Represents an experience of memory.
  """
  pass


class ExperienceWithValues(collections.namedtuple('Experience', ['state', 'next_state',
                                                                 'action', 'action_value',
                                                                 'value',
                                                                 'reward', 'terminal'])):
  """
  Represents an experience (with values) of memory.
  """
  pass


class Replay(collections.namedtuple('Replay', ['state', 'next_state',
                                               'action', 'action_value',
                                               'reward', 'terminal',
                                               'sequence_length'])):
  """
  Represents a sequence of `Experience`s.
  """
  pass


class ReplayWithValues(collections.namedtuple('Replay', ['state', 'next_state',
                                                         'action', 'action_value',
                                                         'value',
                                                         'reward', 'terminal',
                                                         'sequence_length'])):
  """
  Represents a sequence of `ExperienceWithValues`s.
  """
  pass
