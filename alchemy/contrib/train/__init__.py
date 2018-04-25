# -*- coding: utf-8 -*-
from __future__ import absolute_import

from alchemy.contrib.train.sparse import SparseVariableOptimizer
from alchemy.contrib.train.bbb import (
    assign_pruned_by_bbb_to_template,
    prune_by_bbb,
    assign_template_to_prune_by_bbb)
