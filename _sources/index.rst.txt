.. image:: https://github.com/wenkesj/alchemy/blob/master/docs/chem.svg
   :align: center


Alchemy is a utility and contrib extension to the base TensorFlow library.
It includes `notebooks <https://github.com/wenkesj/alchemy/blob/master/notebooks>`_
and tests for recent research papers and my own experiments.


Getting started
===============


Installing Alchemy requires pip. After that, you're set.

.. code-block:: bash

   git clone https://github.com/wenkesj/alchemy
   (cd alchemy; pip install -e .)

Check out these `notebooks <https://github.com/wenkesj/alchemy/blob/master/notebooks>`_
that include detailed examples and implementations of recent research and experiments. All notebooks
were created with `Google Colab <colab.research.google.com>`_ and can be opened and edited
freely!:


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Data <alchemy.contrib.data>
   Abstract Layers <alchemy.contrib.layers>
   Reinforcement Learning <alchemy.contrib.rl>
   Recurrent Neural Networks <alchemy.contrib.rnn>
   Training <alchemy.contrib.train>
   Multiprocessing <alchemy.multiprocessing>
   Utility <alchemy.utils>
