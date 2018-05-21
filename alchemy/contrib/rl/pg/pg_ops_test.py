# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gym
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import training_util
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import adam

from alchemy.utils import distribution_utils
from alchemy.contrib.rl import dataset
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl.pg import pg_ops
from alchemy.contrib.gym import gym_ops
from alchemy.contrib.gym import gym_test_utils
from alchemy.contrib.distributions import sampling_ops


def mlp(inputs, hidden_layers):
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden, units=hidden_size, use_bias=False, activation=nn_ops.relu)
  return hidden


class PGTest(test.TestCase):

  hparams = hparam.HParams(
      learning_rate=1.25e-3,
      hidden_layers=[16, 16],
      initial_exploration=.5,
      use_dropout_exploration=False,
      discount=.99,
      exploration_decay_steps=64,
      exploration_decay_rate=.99,
      entropy_coeff=.01,
      max_sequence_length=33,
      num_episodes=32,
      batch_size=16,
      num_iterations=100)

  def test_pg_ops_advantage(self):
    ops.reset_default_graph()
    np.random.seed(42)
    random_seed.set_random_seed(42)
    env = gym.make('CartPole-v0')
    env.seed(42)

    # Setup the policy and model
    global_step = training_util.get_or_create_global_step()
    deterministic_ph = array_ops.placeholder(
        dtypes.bool, [], name='deterministic')
    exploration_op = learning_rate_decay.exponential_decay(
        PGTest.hparams.initial_exploration,
        global_step,
        PGTest.hparams.exploration_decay_steps,
        PGTest.hparams.exploration_decay_rate)

    state_distribution, state_ph = gym_ops.distribution_from_gym_space(
        env.observation_space, name='state_space')

    with variable_scope.variable_scope('logits'):
      action_value_op = mlp(state_ph, PGTest.hparams.hidden_layers)
      if PGTest.hparams.use_dropout_exploration:
        action_value_op = core.Dropout(exploration_op)(
            action_value_op, gen_math_ops.logical_not(deterministic_ph))

    action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
        env.action_space, logits=[action_value_op], name='action_space')

    if PGTest.hparams.use_dropout_exploration:
      action_op = array_ops.squeeze(action_distribution.mode())
    else:
      action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          action_distribution, exploration_op, deterministic_ph))

    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=PGTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(PGTest.hparams.batch_size)
    replay_op = replay_dataset.make_one_shot_iterator().get_next()

    action_ph = array_ops.placeholder(
        stream.action_dtype, [None, None] + stream.action_shape, name='action')
    reward_ph = array_ops.placeholder(
        stream.reward_dtype, [None, None] + stream.reward_shape, name='reward')
    terminal_ph = array_ops.placeholder(
        dtypes.bool, [None, None], name='terminal')
    sequence_length_ph = array_ops.placeholder(
        dtypes.int32, [None, 1], name='sequence_length')
    sequence_length = array_ops.squeeze(sequence_length_ph, -1)

    # Setup the loss/optimization procedure
    advantage_op = pg_ops.advantage(
        reward_ph,
        sequence_length=sequence_length,
        max_sequence_length=PGTest.hparams.max_sequence_length,
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=PGTest.hparams.discount)

    loss_op = -action_distribution.log_prob(action_ph) * advantage_op
    loss_op += -action_distribution.entropy(name='entropy') * PGTest.hparams.entropy_coeff
    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            sequence_length, loss_op.dtype))
    optimizer = adam.AdamOptimizer(
        learning_rate=PGTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      for iteration in range(PGTest.hparams.num_iterations):

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=PGTest.hparams.num_episodes,
            stream=stream)

        while True:
          try:
            replay = sess.run(replay_op)
          except (errors_impl.InvalidArgumentError, errors_impl.OutOfRangeError):
            break
          _, loss = sess.run(
              (train_op, loss_op),
              feed_dict={
                state_ph: replay.state,
                action_ph: replay.action,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
                deterministic_ph: True,
              })

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=PGTest.hparams.num_episodes,
            deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / PGTest.hparams.num_episodes))


if __name__ == '__main__':
  test.main()
