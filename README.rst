.. image:: https://github.com/wenkesj/alchemy/blob/experimental/docs/chem.svg
    :align: center

Alchemy is a utility and contrib extension to the base TensorFlow library.
It includes `notebooks https://github.com/wenkesj/alchemy/blob/experimental/notebooks`
and tests for recent research papers and my own experiments.

Getting started
===============

Installing Alchemy requires pip. After that, you're set. Check out these
`notebooks https://github.com/wenkesj/alchemy/blob/experimental/notebooks` that include
detailed examples and implementations of recent research and experiments. All notebooks
were created with `Google Colab colab.research.google.com` and can be opened and edited
freely!:

Notebooks
---------

  `Fast Weights https://github.com/wenkesj/alchemy/blob/experimental/notebooks/alchemy_fast_weights.ipynb`
  is a demonstration of the model from the paper
  `Using Fast Weights to Attend to the Recent Past. https://arxiv.org/abs/1610.06258`,
  which is similar to the `tf.contrib.rnn.BasicRNNCell` API. This notebook also includes
  a demonstration of the model from the paper
  `Fast Weight Long Short-Term Memory https://openreview.net/pdf?id=BJI1eDyvz` which is similar
  to the `tf.contrib.rnn.BasicLSTMCell` API. **TL;DR:** Both of these implementations contribute a
  significant advantage over their counterparts for short-term memory (hidden state attention):

  .. image:: https://github.com/wenkesj/alchemy/blob/experimental/docs/fast_weights.png
      :align: center

  `Pruning by Bayes-By-Backprop https://github.com/wenkesj/alchemy/blob/experimental/notebooks/alchemy_pruning.ipynb`
  is a demonstration of the algorithm from the paper
  `Weight Uncertainty in Neural Networks https://arxiv.org/abs/1505.05424`,
  which uses `DeepMind's Sonnet https://github.com/deepmind/sonnet/` API to create a
  `tf.variable_scope` getter, *in vivo*, to transform a variable to a distribution. **TL;DR:** This
  notebook is a live action demonstration of pruning neural networks (one can easily prune
  **> 95%** of the variables) with very little effort:

  .. image:: https://github.com/wenkesj/alchemy/blob/experimental/docs/pruning.png
      :align: center

  `Lateral Connections https://github.com/wenkesj/alchemy/blob/experimental/notebooks/alchemy_lateral.ipynb`
  is an experiment that attempts to model the lateral connections found in neural networks in the
  brain. **TL;DR**: Residual connections can be seen as a special case of lateral, or complex,
  (identity weighted) connections. The connections can be visualized and can possibly reveal
  relationships between neurons.

  .. image:: https://github.com/wenkesj/alchemy/blob/experimental/docs/lateral_connections.png
      :align: center

  .. image:: https://github.com/wenkesj/alchemy/blob/experimental/docs/lateral_connectivity.png
      :align: center

  `Winner-take-all https://github.com/wenkesj/alchemy/blob/experimental/notebooks/alchemy_wta.ipynb`
  is an experiment that attempts to model the inhibiting behavior found in neural networks in the
  brain. It encapsulates the idea of lateral-inhibition and clustering by only allowing the top `k`
  activations and zero-ing out the rest. **TL;DR:** Largely a work-in-progress, very interested in
  the unsupervised capabilities.


Developer install
-----------------

To install the latest version of Alchemy (in development mode):

.. code-block:: bash

  git clone https://github.com/wenkesj/alchemy
  (cd alchemy; pip install -e .)

Library install
---------------

To install the latest PyPI release as a library (in user mode):

.. code-block:: bash

  pip install --user alchemy
