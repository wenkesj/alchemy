# -*- coding: utf-8 -*-
import collections


class Transition(collections.namedtuple('Transition', ['state', 'action', 'values',
                                                       'reward', 'terminal', 'info'])):
  pass


class Trajectory(object):
  def __init__(self, transitions, weight=0.):
    self.transitions = transitions
    self.size = len(transitions)
    self.weight = weight

  def add(self, transition):
    self.transitions.append(transition)
    self.size += 1
    return self

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
