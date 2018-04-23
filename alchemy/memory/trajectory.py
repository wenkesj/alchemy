# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections


# TODO(wenkesj): add docstring
class Transition(collections.namedtuple('Transition', ['state', 'action', 'values',
                                                       'reward', 'terminal', 'info'])):
  pass


# TODO(wenkesj): add docstring
class Trajectory(object):

  def __init__(self, transitions, weight=0.):
    self.transitions = transitions
    self.size = len(transitions)
    self.weight = weight

  def add(self, transition):
    self.transitions.append(transition)
    self.size += 1
    return self

  @property
  def state(self):
    return [t.state for t in self.transitions]

  @property
  def action(self):
    return [t.action for t in self.transitions]

  @property
  def values(self):
    return [t.values for t in self.transitions]

  @property
  def reward(self):
    return [t.reward for t in self.transitions]

  @property
  def terminal(self):
    return [t.terminal for t in self.transitions]

  @property
  def info(self):
    return [t.info for t in self.transitions]

  def __gt__(self, x):
    return self.weight > x.weight

  def __lt__(self, x):
    return self.weight < x.weight

  def __eq__(self, x):
    return self.weight == x.weight

  def __len__(self):
    return self.size

  def __repr__(self):
    return '<Trajectory(weight={})>'.format(self.weight)
