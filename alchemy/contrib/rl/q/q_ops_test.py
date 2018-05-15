# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gym
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.layers import core

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import adam

from alchemy.utils import distribution_utils
from alchemy.utils import shortcuts
from alchemy.utils import sequence_utils
from alchemy.contrib.losses import losses_impl
from alchemy.contrib.rl import dataset
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl.q import q_ops
from alchemy.contrib.gym import gym_ops
from alchemy.contrib.gym import gym_test_utils
from alchemy.contrib.distributions import sampling_ops


def mlp(inputs, hidden_layers):
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden, units=hidden_size, use_bias=False, activation=nn_ops.relu)
  return hidden


class QTest(test.TestCase):

  hparams = hparam.HParams(
      learning_rate=1.25e-3,
      hidden_layers=[8],
      initial_exploration=.5,
      discount=.99,
      exploration_decay_steps=256 // 16 * 25,
      exploration_decay_rate=.99,
      max_sequence_length=1,
      num_episodes=256,
      batch_size=16,
      num_iterations=100,
      assign_target_steps=10 * 16,
      huber_loss_delta=1.,
      num_quantiles=51)

  @test_util.skip_if(True)
  def test_q_ops_dqn(self):
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
        QTest.hparams.initial_exploration,
        global_step,
        QTest.hparams.exploration_decay_steps,
        QTest.hparams.exploration_decay_rate)


    state_distribution, state_ph = gym_ops.distribution_from_gym_space(
        env.observation_space, name='state_space')
    with variable_scope.variable_scope('logits'):
      action_value_op = mlp(state_ph, QTest.hparams.hidden_layers)
      action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[action_value_op], name='action_space')
      action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          action_distribution, exploration_op, deterministic_ph))
    policy_variables = variables.trainable_variables(scope='logits')


    next_state_ph = shortcuts.placeholder_like(state_ph, name='next_state_space')
    with variable_scope.variable_scope('logits', reuse=True):
      next_action_value_op = mlp(next_state_ph, QTest.hparams.hidden_layers)
      next_action_distribution, next_action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[next_action_value_op], name='action_space')
      next_action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          next_action_distribution, exploration_op, deterministic_ph))


    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=QTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(QTest.hparams.batch_size)
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

    q_value_op, expected_q_value_op = q_ops.expected_q_value(
        reward_ph,
        action_ph,
        action_value_op,
        next_action_value_op,
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=QTest.hparams.discount)

    # mean_squared_error
    loss_op = math_ops.square(q_value_op - expected_q_value_op)

    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            sequence_length, loss_op.dtype))
    optimizer = adam.AdamOptimizer(
        learning_rate=QTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      for iteration in range(QTest.hparams.num_iterations):
        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
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
                next_state_ph: replay.next_state,
                action_ph: replay.action,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
              })

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
            deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / QTest.hparams.num_episodes))

  # @test_util.skip_if(True)
  def test_q_ops_double_dqn(self):
    env = gym.make('CartPole-v0')
    ops.reset_default_graph()
    np.random.seed(42)
    random_seed.set_random_seed(42)
    env.seed(42)

    # Setup the policy and model
    global_step = training_util.get_or_create_global_step()
    deterministic_ph = array_ops.placeholder(
        dtypes.bool, [], name='deterministic')
    exploration_op = learning_rate_decay.exponential_decay(
        QTest.hparams.initial_exploration,
        global_step,
        QTest.hparams.exploration_decay_steps,
        QTest.hparams.exploration_decay_rate)


    state_distribution, state_ph = gym_ops.distribution_from_gym_space(
        env.observation_space, name='state_space')
    with variable_scope.variable_scope('logits'):
      action_value_op = mlp(state_ph, QTest.hparams.hidden_layers)
      action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[action_value_op], name='action_space')
      action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          action_distribution, exploration_op, deterministic_ph))
    policy_variables = variables.trainable_variables(scope='logits')


    next_state_ph = shortcuts.placeholder_like(state_ph, name='next_state_space')
    with variable_scope.variable_scope('logits', reuse=True):
      next_action_value_op = mlp(next_state_ph, QTest.hparams.hidden_layers)
      next_action_distribution, next_action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[next_action_value_op], name='action_space')
      next_action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          next_action_distribution, exploration_op, deterministic_ph))

    with variable_scope.variable_scope('target_logits'):
      target_next_action_value_op = mlp(next_state_ph, QTest.hparams.hidden_layers)
      target_next_action_distribution, target_next_action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[target_next_action_value_op], name='action_space')
      target_next_action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          target_next_action_distribution, exploration_op, deterministic_ph))
    assign_target_op = shortcuts.assign_scope('logits', 'target_logits')


    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=QTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(QTest.hparams.batch_size)
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

    q_value_op, expected_q_value_op = q_ops.expected_q_value(
        reward_ph,
        action_ph,
        action_value_op,
        (next_action_value_op, target_next_action_value_op),
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=QTest.hparams.discount)

    # mean_squared_error
    loss_op = math_ops.square(q_value_op - expected_q_value_op)
    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            sequence_length, loss_op.dtype))
    optimizer = adam.AdamOptimizer(
        learning_rate=QTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)
    train_op = control_flow_ops.cond(
        gen_math_ops.equal(
            gen_math_ops.mod(
                ops.convert_to_tensor(
                    QTest.hparams.assign_target_steps, dtype=dtypes.int64),
                (global_step + 1)), 0),
        lambda: control_flow_ops.group(*[train_op, assign_target_op]),
        lambda: train_op)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_target_op)

      for iteration in range(QTest.hparams.num_iterations):
        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
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
                next_state_ph: replay.next_state,
                action_ph: replay.action,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
              })

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
            deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / QTest.hparams.num_episodes))

  @test_util.skip_if(True)
  def test_q_ops_quantile_dqn(self):
    env = gym.make('CartPole-v0')
    ops.reset_default_graph()
    np.random.seed(42)
    random_seed.set_random_seed(42)
    env.seed(42)

    # Setup the policy and model
    global_step = training_util.get_or_create_global_step()
    deterministic_ph = array_ops.placeholder(
        dtypes.bool, [], name='deterministic')
    exploration_op = learning_rate_decay.exponential_decay(
        QTest.hparams.initial_exploration,
        global_step,
        QTest.hparams.exploration_decay_steps,
        QTest.hparams.exploration_decay_rate)

    state_distribution, state_ph = gym_ops.distribution_from_gym_space(
        env.observation_space, name='state_space')
    action_distribution, _ = gym_ops.distribution_from_gym_space(
        env.action_space, name='action_space')

    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution)

    with variable_scope.variable_scope('logits'):
      action_value_op = mlp(state_ph, QTest.hparams.hidden_layers)
      action_value_op = core.dense(
          action_value_op,
          stream.action_value_shape[-1] * QTest.hparams.num_quantiles,
          use_bias=False)
      action_value_op_shape = array_ops.shape(action_value_op)
      action_value_shape = [
          action_value_op_shape[0],
          action_value_op_shape[1],
          stream.action_value_shape[-1],
          QTest.hparams.num_quantiles]
      action_value_op = gen_array_ops.reshape(action_value_op, action_value_shape)
      mean_action_value_op = math_ops.reduce_mean(action_value_op, axis=-1)
      action_op = math_ops.argmax( mean_action_value_op, axis=-1)
      action_op = array_ops.squeeze(action_op)
    policy_variables = variables.trainable_variables(scope='logits')


    next_state_ph = shortcuts.placeholder_like(state_ph, name='next_state_space')
    with variable_scope.variable_scope('targets'):
      target_next_action_value_op = mlp(next_state_ph, QTest.hparams.hidden_layers)
      target_next_action_value_op = core.dense(
          target_next_action_value_op,
          stream.action_value_shape[-1] * QTest.hparams.num_quantiles,
          use_bias=False)
      target_next_action_value_op_shape = array_ops.shape(target_next_action_value_op)
      target_next_action_value_shape = [
          target_next_action_value_op_shape[0],
          target_next_action_value_op_shape[1],
          stream.action_value_shape[-1],
          QTest.hparams.num_quantiles]
      target_next_action_value_op = gen_array_ops.reshape(
          target_next_action_value_op, target_next_action_value_shape)
      mean_target_next_action_value_op = math_ops.reduce_mean(
          target_next_action_value_op, axis=-1)
    assign_target_op = shortcuts.assign_scope('logits', 'target_logits')


    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=QTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(QTest.hparams.batch_size)
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

    q_value_op, expected_q_value_op = q_ops.expected_q_value(
        array_ops.expand_dims(reward_ph, -1),
        action_ph,
        action_value_op,
        (target_next_action_value_op, mean_target_next_action_value_op),
        weights=array_ops.expand_dims(
            1 - math_ops.cast(terminal_ph, reward_ph.dtype), -1),
        discount=QTest.hparams.discount)

    u = expected_q_value_op - q_value_op
    loss_op = losses_impl.huber_loss(u, delta=QTest.hparams.huber_loss_delta)

    tau_op = (2. * math_ops.range(
        0, QTest.hparams.num_quantiles, dtype=u.dtype) + 1) / (
            2. * QTest.hparams.num_quantiles)

    loss_op *= math_ops.abs(tau_op - math_ops.cast(u < 0, tau_op.dtype))

    loss_op = math_ops.reduce_mean(loss_op, axis=-1)
    # loss_op = math_ops.reduce_mean(loss_op, axis=-2)
    # loss_op = math_ops.reduce_sum(loss_op, axis=-1)

    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            sequence_length, loss_op.dtype))
    optimizer = adam.AdamOptimizer(
        learning_rate=QTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)
    train_op = control_flow_ops.cond(
        gen_math_ops.equal(
            gen_math_ops.mod(
                ops.convert_to_tensor(
                    QTest.hparams.assign_target_steps, dtype=dtypes.int64),
                (global_step + 1)), 0),
        lambda: control_flow_ops.group(*[train_op, assign_target_op]),
        lambda: train_op)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_target_op)

      for iteration in range(QTest.hparams.num_iterations):
        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            mean_action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
            stream=stream)

        while True:
          try:
            replay = sess.run(replay_op)
          except (errors_impl.InvalidArgumentError, errors_impl.OutOfRangeError):
            break
          loss, _ = sess.run(
              (loss_op, train_op),
              feed_dict={
                state_ph: replay.state,
                next_state_ph: replay.next_state,
                action_ph: replay.action,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
              })

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            mean_action_value_op, action_op,
            num_episodes=QTest.hparams.num_episodes,
            deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / QTest.hparams.num_episodes))


if __name__ == '__main__':
  test.main()
