# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import string

from alchemy.utils import assert_utils


# TODO(wenkesj): add docstring
# TODO(wenkesj): make this tf.data.Dataset compatible.
class ART(object):

  """
  Associative retrieval task (ART + mART)
  https://arxiv.org/abs/1610.06258
  """

  def __init__(self, chars=list(string.ascii_lowercase)):
    self._chars = chars
    self._chars_size = len(self._chars) - 1
    self._alphabet = self._chars + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '?']
    self._alphabet_size = len(self._alphabet) - 1
    self._encoder = np.eye(self._alphabet_size + 1)

  @property
  def vocab_size(self):
    return len(self._encoder)

  def ordinal_to_alpha(self, sequence):
    conversion = ""
    for item in sequence:
      conversion += str(self._alphabet[int(item)])
    return conversion

  def create_example(self, k=8, use_modified=False):
    q, r = divmod(k, 2)
    assert_utils.assert_true(
        r == 0 and k > 1 and k < self._alphabet_size,
        "k must be even, > 1, and < {}".format(self._alphabet_size))

    letters = np.random.choice(range(0, self._chars_size), q, replace=False)
    numbers = np.random.choice(
        range(self._chars_size + 1, self._alphabet_size), q, replace=True)
    if use_modified:
      x = np.concatenate((letters, numbers))
    else:
      x = np.stack((letters, numbers)).T.ravel()

    x = np.append(x, [self._alphabet_size, self._alphabet_size])
    index = np.random.choice(range(0, q), 1, replace=False)
    x = np.append(x, [letters[index]]).astype('int')
    y = numbers[index]
    return self._encoder[x], self._encoder[y][0]
