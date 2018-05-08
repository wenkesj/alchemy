# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import string

from alchemy.utils import assert_utils


class ART(object):

  """
  Implements the associative retrieval task (ART + mART)
  https://arxiv.org/abs/1610.06258
  """

  def __init__(self, chars=list(string.ascii_lowercase)):
    """Create a new ART instance that creates samples from the alphabet `chars`.

    Arguments:
      chars: `list` of `str` that represents the alphabet to sample from. Must not include
          numbers/characters [0-9] or '?'.
    """
    self._chars = chars
    self._chars_size = len(self._chars) - 1
    self._alphabet = self._chars + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '?']
    self._alphabet_size = len(self._alphabet) - 1
    self._encoder = np.eye(self._alphabet_size + 1)

  @property
  def vocab_size(self):
    """Returns the sample space vocab size (i.e. len(chars) + [0-9] + '?'))"""
    return len(self._encoder)

  def ordinal_to_alpha(self, sequence):
    """Convert a sequence of label-encoded alpha-numerics back to alpha-numerics.

    Arguments:
      sequence: 1D iterable containing the labels.

    Returns:
      1D iterable containing the alpha-numerics.
    """
    conversion = ""
    for item in sequence:
      conversion += str(self._alphabet[int(item)])
    return conversion

  def create_example(self, k=8, use_modified=False):
    """Creates a single (modified) example of length `k`.

    Arguments:
      k: an even `int` that defines the length of the sample space. For example, if `k = 8` and the
          vocab contains `ATCG`, then a sample would look like this: (A9C5G1T3??C, 5).
      use_modified: `bool` that, when `True`, makes samples contiguous alpha-numeric. For example,
          when `k = 8` and the vocab contains `ATCG`, then a sample would look like this:
          (ACTG9513??C, 5). The label is the same as the unmodified version, but the sequence is no
          longer zipped.

    Returns:
      A tuple containing the one-hot encoded values from the vocab. For example, if `k = 8` and the
          vocab contains `ATCG`, then this would return onehot(A9C5G1T3??C, 5), where the
          length of the onehot encoding vectors = len('ACTG') + len([0...9]) + len('?')
          = `vocab_size`.
    """
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
    return (self._encoder[x], self._encoder[y][0])
