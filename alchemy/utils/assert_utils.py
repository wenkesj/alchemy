# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numbers

def assert_true(cond, msg='Failed assertion'):
  if not cond:
    raise AssertionError(msg)

def assert_false(cond, msg='Failed assertion'):
  if cond:
    raise AssertionError(msg)

def is_iterable(x):
  """Return a `True` if `x` is iterable."""
  try:
    iter(x)
    return True
  except TypeError:
    return False
  finally:
    return True

def is_number(x):
  """Return a `True` if `x` is a number."""
  return isinstance(x, numbers.Number)
