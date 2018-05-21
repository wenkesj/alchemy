# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gym
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
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
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import training_util
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import adam

from alchemy.utils import distribution_utils
from alchemy.utils import sequence_utils
from alchemy.contrib.rnn import stacked_rnn_impl
from alchemy.contrib.rl import dataset
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl.pg import pg_ops
from alchemy.contrib.gym import gym_ops
from alchemy.contrib.gym import gym_test_utils
from alchemy.contrib.gym.envs import bandits
from alchemy.contrib.distributions import sampling_ops


def mlp(inputs, hidden_layers):
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden, units=hidden_size, use_bias=False, activation=nn_ops.relu)
  return hidden

def rnn(inputs, hidden_layers):
  outputs, states, initial_states, zero_state = stacked_rnn_impl.stacked_rnn(
      inputs, hidden_layers, rnn_cell_impl.BasicLSTMCell)
  return outputs, states, initial_states, zero_state



class PGTest(test.TestCase):

  hparams = hparam.HParams(
      learning_rate=1.25e-3,
      hidden_layers=[16, 16],
      initial_exploration=.5,
      use_dropout_exploration=False,
      discount=.8,
      lambda_td=1.,
      epsilon=.2,
      exploration_decay_steps=64,
      exploration_decay_rate=.99,
      entropy_coeff=.01,
      value_coeff=.5,
      value_units=16,
      assign_policy_steps=16,
      max_sequence_length=33,
      max_meta_sequence_length=99,
      num_episodes=32,
      num_meta_train_episodes=32,
      num_meta_test_episodes=10,
      batch_size=16,
      replay_epochs=2,
      num_iterations=100)

  @test_util.skip_if(True)
  def test_pg_ops_advantage(self):
    """This tests the PG algorithm with baseline advantage estimation."""
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

  @test_util.skip_if(True)
  def test_pg_ops_generalized_advantage_estimate(self):
    """This tests the PPO algorithm with GAE.

    Reference:
      https://arxiv.org/abs/1707.06347
    """
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

    # values
    with variable_scope.variable_scope('logits') as vs:
      state_body_op = mlp(state_ph, PGTest.hparams.hidden_layers)
      if PGTest.hparams.use_dropout_exploration:
        state_body_op = core.Dropout(exploration_op)(
            state_body_op, gen_math_ops.logical_not(deterministic_ph))
      action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[state_body_op], name='action_space')
      if PGTest.hparams.use_dropout_exploration:
        action_op = array_ops.squeeze(action_distribution.mode())
      else:
        action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
            action_distribution, exploration_op, deterministic_ph))
      value_body_op = core.dense(
          state_body_op, units=PGTest.hparams.value_units,
          activation=nn_ops.relu, use_bias=False)
      value_op = array_ops.squeeze(core.dense(value_body_op, units=1, use_bias=False), -1)
      policy_variables = ops.get_collection(
          ops.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

    # target
    with variable_scope.variable_scope('old_logits') as vs:
      old_state_body_op = mlp(state_ph, PGTest.hparams.hidden_layers)
      if PGTest.hparams.use_dropout_exploration:
        old_state_body_op = core.Dropout(exploration_op)(
            old_state_body_op, gen_math_ops.logical_not(deterministic_ph))
      old_action_distribution, old_action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[old_state_body_op], name='action_space')
      if PGTest.hparams.use_dropout_exploration:
        old_action_op = array_ops.squeeze(old_action_distribution.mode())
      else:
        old_action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
            old_action_distribution, exploration_op, deterministic_ph))
      old_value_body_op = core.dense(
          old_state_body_op, units=PGTest.hparams.value_units,
          activation=nn_ops.relu, use_bias=False)
      old_value_op = array_ops.squeeze(core.dense(old_value_body_op, units=1, use_bias=False), -1)
      old_policy_variables = ops.get_collection(
          ops.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)


    assign_policy_op = control_flow_ops.group(*list(state_ops.assign(t, f)
                                                    for f, t in zip(
                                                        policy_variables, old_policy_variables)))

    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution,
        with_values=True)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=PGTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(PGTest.hparams.batch_size)
    replay_dataset = replay_dataset.repeat(PGTest.hparams.replay_epochs)
    replay_op = replay_dataset.make_one_shot_iterator().get_next()


    action_ph = array_ops.placeholder(
        stream.action_dtype,
        [None, None] + stream.action_shape,
        name='action')
    value_ph = array_ops.placeholder(
        stream.reward_dtype,
        [None, None] + stream.reward_shape,
        name='value')
    reward_ph = array_ops.placeholder(
        stream.reward_dtype,
        [None, None] + stream.reward_shape,
        name='reward')
    terminal_ph = array_ops.placeholder(
        dtypes.bool,
        [None, None],
        name='terminal')
    sequence_length_ph = array_ops.placeholder(
        dtypes.int32,
        [None, 1],
        name='sequence_length')
    sequence_length = array_ops.squeeze(sequence_length_ph, -1)


    # Setup the loss/optimization procedure
    advantage_op, return_op = pg_ops.generalized_advantage_estimate(
        reward_ph,
        value_ph,
        sequence_length,
        max_sequence_length=PGTest.hparams.max_sequence_length,
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=PGTest.hparams.discount,
        lambda_td=PGTest.hparams.lambda_td)

    # actor loss
    logits_prob = -action_distribution.log_prob(action_ph)
    old_logits_prob = -old_action_distribution.log_prob(action_ph)
    ratio = math_ops.exp(logits_prob - old_logits_prob)
    clipped_ratio = clip_ops.clip_by_value(
        ratio, 1. - PGTest.hparams.epsilon, 1. + PGTest.hparams.epsilon)

    actor_loss_op = -math_ops.minimum(ratio * advantage_op, clipped_ratio * advantage_op)
    critic_loss_op = math_ops.square(return_op - value_op) * PGTest.hparams.value_coeff
    entropy_loss_op = -action_distribution.entropy(name='entropy') * PGTest.hparams.entropy_coeff
    loss_op = entropy_loss_op - actor_loss_op - critic_loss_op
    loss_op = -math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, -1) / math_ops.cast(
                sequence_length, logits_prob.dtype))

    optimizer = adam.AdamOptimizer(
        learning_rate=PGTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_policy_op)

      for iteration in range(PGTest.hparams.num_iterations):
        rewards = gym_test_utils.rollout_with_values_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            old_action_value_op, old_action_op, old_value_op,
            num_episodes=PGTest.hparams.num_episodes,
            stream=stream)

        sess.run(assign_policy_op)
        while True:
          try:
            replay = sess.run(replay_op)
          except (errors_impl.InvalidArgumentError, errors_impl.OutOfRangeError) as e:
            break
          _, loss = sess.run(
              (train_op, loss_op),
              feed_dict={
                state_ph: replay.state,
                action_ph: replay.action,
                value_ph: replay.value,
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

  # @test_util.skip_if(True)
  def test_pg_ops_meta_rl(self):
    """This tests the PPO algorithm with GAE for meta-RL on the dependent bandits tasks.

    Reference:
      https://arxiv.org/abs/1611.05763
    """

    ops.reset_default_graph()
    np.random.seed(42)
    random_seed.set_random_seed(42)

    train_envs = {
      'hard': bandits.BanditHard(),
      'medium': bandits.BanditMedium(),
    }
    test_envs = {
      'easy': bandits.BanditEasy(),
    }

    for env in train_envs.values():
      env.seed(42)
    for env in test_envs.values():
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
        train_envs['medium'].observation_space, name='state_space')

    # TODO(wenkesj): make this not necessary, for now it's okay...
    dummy_action_distribution, _ = gym_ops.distribution_from_gym_space(
        train_envs['medium'].action_space, name='dummy_action_space')

    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, dummy_action_distribution,
        with_values=True)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=PGTest.hparams.max_meta_sequence_length)
    replay_dataset = replay_dataset.batch(PGTest.hparams.batch_size)
    replay_dataset = replay_dataset.repeat(PGTest.hparams.replay_epochs)
    replay_op = replay_dataset.make_one_shot_iterator().get_next()

    action_ph = array_ops.placeholder(
        stream.action_dtype,
        [None, None] + stream.action_shape,
        name='action')
    value_ph = array_ops.placeholder(
        stream.reward_dtype,
        [None, None] + stream.reward_shape,
        name='value')
    reward_ph = array_ops.placeholder(
        stream.reward_dtype,
        [None, None] + stream.reward_shape,
        name='reward')
    terminal_ph = array_ops.placeholder(
        dtypes.bool,
        [None, None],
        name='terminal')
    sequence_length_ph = array_ops.placeholder(
        dtypes.int32,
        [None, 1],
        name='sequence_length')
    sequence_length = array_ops.squeeze(sequence_length_ph, -1)
    previous_action = sequence_utils.shift(action_ph, axis=-1, rotations=1)
    previous_reward = array_ops.expand_dims(
        sequence_utils.shift(reward_ph, axis=-1, rotations=1), -1)

    # TODO(wenkesj): make this compatible with other distributions
    previous_action = array_ops.one_hot(
        previous_action, stream.action_value_shape[-1],
        dtype=stream.action_value_dtype)
    initial_action = 0

    state_in = array_ops.concat(
        [math_ops.cast(state_ph, dtypes.float32),
         previous_action,
         previous_reward],
        axis=-1)

    # values
    with variable_scope.variable_scope('logits') as vs:
      state_body_op, internal_state_op, internal_state_ph, zero_state_fn = rnn(
          state_in, PGTest.hparams.hidden_layers)

      if PGTest.hparams.use_dropout_exploration:
        state_body_op = core.Dropout(exploration_op)(
            state_body_op, gen_math_ops.logical_not(deterministic_ph))

      action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
          train_envs['medium'].action_space, logits=[state_body_op], name='action_space')

      if PGTest.hparams.use_dropout_exploration:
        action_op = array_ops.squeeze(action_distribution.mode())
      else:
        action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
            action_distribution, exploration_op, deterministic_ph))
      value_body_op = core.dense(
          state_body_op, units=PGTest.hparams.value_units,
          activation=nn_ops.relu, use_bias=False)
      value_op = array_ops.squeeze(core.dense(value_body_op, units=1, use_bias=False), -1)
      policy_variables = ops.get_collection(
          ops.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

    # target
    with variable_scope.variable_scope('old_logits') as vs:
      old_state_body_op, old_internal_state_op, old_internal_state_ph, old_zero_state_fn = rnn(
          state_in, PGTest.hparams.hidden_layers)

      if PGTest.hparams.use_dropout_exploration:
        old_state_body_op = core.Dropout(exploration_op)(
            old_state_body_op, gen_math_ops.logical_not(deterministic_ph))

      old_action_distribution, old_action_value_op = gym_ops.distribution_from_gym_space(
          train_envs['medium'].action_space, logits=[old_state_body_op], name='action_space')

      if PGTest.hparams.use_dropout_exploration:
        old_action_op = array_ops.squeeze(old_action_distribution.mode())
      else:
        old_action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
            old_action_distribution, exploration_op, deterministic_ph))
      old_value_body_op = core.dense(
          old_state_body_op, units=PGTest.hparams.value_units,
          activation=nn_ops.relu, use_bias=False)
      old_value_op = array_ops.squeeze(
          core.dense(old_value_body_op, units=1, use_bias=False), -1)
      old_policy_variables = ops.get_collection(
          ops.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)


    assign_policy_op = control_flow_ops.group(*list(state_ops.assign(t, f)
                                                    for f, t in zip(
                                                        policy_variables, old_policy_variables)))


    # Setup the loss/optimization procedure
    advantage_op, return_op = pg_ops.generalized_advantage_estimate(
        reward_ph,
        value_ph,
        sequence_length,
        max_sequence_length=PGTest.hparams.max_meta_sequence_length,
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=PGTest.hparams.discount,
        lambda_td=PGTest.hparams.lambda_td)

    # actor loss
    logits_prob = -action_distribution.log_prob(action_ph)
    old_logits_prob = -old_action_distribution.log_prob(action_ph)
    ratio = math_ops.exp(logits_prob - old_logits_prob)
    clipped_ratio = clip_ops.clip_by_value(
        ratio, 1. - PGTest.hparams.epsilon, 1. + PGTest.hparams.epsilon)

    actor_loss_op = -math_ops.minimum(ratio * advantage_op, clipped_ratio * advantage_op)
    critic_loss_op = math_ops.square(return_op - value_op) * PGTest.hparams.value_coeff
    entropy_loss_op = -action_distribution.entropy(name='entropy') * PGTest.hparams.entropy_coeff
    loss_op = entropy_loss_op - actor_loss_op - critic_loss_op
    loss_op = -math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, -1) / math_ops.cast(
                sequence_length, logits_prob.dtype))

    optimizer = adam.AdamOptimizer(
        learning_rate=PGTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_policy_op)

      for iteration in range(PGTest.hparams.num_iterations):
        for train_name, train_env in train_envs.items():
          rewards = gym_test_utils.rollout_meta_with_values_on_gym_env(
              sess, train_env, state_ph, old_internal_state_ph,
              action_ph, reward_ph, deterministic_ph,
              old_action_value_op, old_action_op,
              old_value_op, old_internal_state_op,
              old_zero_state_fn, initial_action,
              num_episodes=PGTest.hparams.num_meta_train_episodes,
              stream=stream)

        sess.run(assign_policy_op)
        while True:
          try:
            replay = sess.run(replay_op)
          except (errors_impl.InvalidArgumentError, errors_impl.OutOfRangeError) as e:
            break

          internal_state = sess.run(zero_state_fn(replay.state.shape[0], dtype=state_ph.dtype))
          _, loss = sess.run(
              (train_op, loss_op),
              feed_dict={
                state_ph: replay.state,
                action_ph: replay.action,
                value_ph: replay.value,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
                **{k: v for k, v in zip(internal_state_ph, internal_state)},
                **{k: v for k, v in zip(old_internal_state_ph, internal_state)},
                deterministic_ph: True,
              })

        for test_env_name, test_env in test_envs.items():
          rewards = gym_test_utils.rollout_meta_on_gym_env(
              sess, test_env, state_ph, internal_state_ph,
              action_ph, reward_ph, deterministic_ph,
              action_value_op, action_op, internal_state_op,
              zero_state_fn, initial_action,
              num_episodes=PGTest.hparams.num_meta_test_episodes,
              deterministic=True, save_replay=False)
          print('{}: average_rewards = {}'.format(
              test_env_name, rewards / PGTest.hparams.num_meta_test_episodes))


if __name__ == '__main__':
  test.main()
