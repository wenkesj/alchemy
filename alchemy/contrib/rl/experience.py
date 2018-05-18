# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections

from alchemy.utils import array_utils


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

  def partition(self, limit, allow_overflow=True):
    state, next_state, action, action_value, reward, terminal, sequence_length = self

    partitions = array_utils.partition(
        tuple(zip(state, next_state, action, action_value, reward, terminal)),
        limit, allow_overflow=allow_overflow)

    replay_partitions = []
    for partition in partitions:
      state, next_state, action, action_value, reward, terminal = partition
      replay_partitions.append(
          Replay(
              state=state,
              next_state=next_state,
              action=action,
              action_value=action_value,
              reward=reward,
              terminal=terminal,
              sequence_length=len(state)))
    return replay_partitions


class ReplayWithValues(collections.namedtuple('Replay', ['state', 'next_state',
                                                         'action', 'action_value',
                                                         'value',
                                                         'reward', 'terminal',
                                                         'sequence_length'])):
  """
  Represents a sequence of `ExperienceWithValues`s.
  """
  def partition(self, limit, allow_overflow=True):
    state, next_state, action, action_value, value, reward, terminal, sequence_length = self

    partitions = array_utils.partition(
        tuple(zip(state, next_state, action, action_value, value, reward, terminal)),
        limit, allow_overflow=allow_overflow)

    replay_partitions = []
    for partition in partitions:
      state, next_state, action, action_value, value, reward, terminal = partition
      replay_partitions.append(
          ReplayWithValues(
              state=state,
              next_state=next_state,
              action=action,
              action_value=action_value,
              value=value,
              reward=reward,
              terminal=terminal,
              sequence_length=len(state)))
    return replay_partitions
