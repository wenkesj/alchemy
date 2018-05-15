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
from tensorflow.python.ops import clip_ops
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
from alchemy.utils import shortcuts
from alchemy.contrib.rl import dataset
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl.ppo import ppo_ops
from alchemy.contrib.gym import gym_ops
from alchemy.contrib.gym import gym_test_utils
from alchemy.contrib.distributions import sampling_ops


def mlp(inputs, hidden_layers):
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden,
        units=hidden_size,
        use_bias=False,
        activation=nn_ops.relu)
  return hidden


class PPOTest(test.TestCase):

  hparams = hparam.HParams(
      learning_rate=1.25e-3,
      hidden_layers=[16, 16],
      initial_exploration=.5,
      discount=.99,
      epsilon=.2,
      lambda_td=1.,
      exploration_decay_steps=256 // 16 * 25,
      exploration_decay_rate=.99,
      entropy_coeff=.01,
      value_coeff=1.,
      assign_policy_steps=64,
      max_sequence_length=200,
      num_episodes=256,
      batch_size=16,
      num_iterations=100)

  def test_ppo_ops_gae(self):
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
        PPOTest.hparams.initial_exploration,
        global_step,
        PPOTest.hparams.exploration_decay_steps,
        PPOTest.hparams.exploration_decay_rate)

    state_distribution, state_ph = gym_ops.distribution_from_gym_space(
        env.observation_space, name='state_space')

    # values
    with variable_scope.variable_scope('logits'):
      body_op = mlp(state_ph, PPOTest.hparams.hidden_layers)
      action_distribution, action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space, logits=[body_op], name='action_space')
      action_op = array_ops.squeeze(sampling_ops.epsilon_greedy(
          action_distribution, exploration_op, deterministic_ph))
      value_op = array_ops.squeeze(core.dense(body_op, units=1, use_bias=False), -1)
    policy_variables = variables.trainable_variables(scope='logits')

    # target
    with variable_scope.variable_scope('old_logits'):
      old_body_op = mlp(state_ph, PPOTest.hparams.hidden_layers)
      old_action_distribution, old_action_value_op = gym_ops.distribution_from_gym_space(
          env.action_space,
          logits=[old_body_op],
          name='action_space')
    assign_policy_op = shortcuts.assign_scope('logits', 'old_logits')


    # Setup the dataset
    stream = streams.Uniform.from_distributions(
        state_distribution, action_distribution,
        with_values=True)
    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=PPOTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(PPOTest.hparams.batch_size)
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
    advantage_op, return_op = ppo_ops.generalized_advantage_estimate(
        reward_ph,
        value_ph,
        sequence_length,
        max_sequence_length=PPOTest.hparams.max_sequence_length,
        weights=(1 - math_ops.cast(terminal_ph, reward_ph.dtype)),
        discount=PPOTest.hparams.discount,
        lambda_td=PPOTest.hparams.lambda_td)

    # actor loss
    logits_prob = action_distribution.log_prob(action_ph)
    old_logits_prob = old_action_distribution.log_prob(action_ph)
    ratio = math_ops.exp(logits_prob - old_logits_prob)
    clipped_ratio = clip_ops.clip_by_value(
        ratio, 1. - PPOTest.hparams.epsilon, 1. + PPOTest.hparams.epsilon)
    actor_loss_op = math_ops.minimum(ratio * advantage_op, clipped_ratio * advantage_op)
    critic_loss_op = math_ops.square(return_op - value_op) * PPOTest.hparams.value_coeff
    entropy_loss_op = -action_distribution.entropy(name='entropy') * PPOTest.hparams.entropy_coeff
    loss_op = actor_loss_op - critic_loss_op + entropy_loss_op
    loss_op = -loss_op

    # total loss
    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            sequence_length, loss_op.dtype))

    optimizer = adam.AdamOptimizer(
        learning_rate=PPOTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=policy_variables)
    train_op = control_flow_ops.cond(
        gen_math_ops.equal(
            gen_math_ops.mod(
                ops.convert_to_tensor(
                    PPOTest.hparams.assign_policy_steps, dtype=dtypes.int64),
                (global_step + 1)), 0),
        lambda: control_flow_ops.group(*[train_op, assign_policy_op]),
        lambda: train_op)


    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(assign_policy_op)

      for iteration in range(PPOTest.hparams.num_iterations):
        rewards = gym_test_utils.rollout_with_values_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op, value_op,
            num_episodes=PPOTest.hparams.num_episodes,
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
                value_ph: replay.value,
                reward_ph: replay.reward,
                terminal_ph: replay.terminal,
                sequence_length_ph: replay.sequence_length,
              })

        rewards = gym_test_utils.rollout_on_gym_env(
            sess, env, state_ph, deterministic_ph,
            action_value_op, action_op,
            num_episodes=PPOTest.hparams.num_episodes,
            deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / PPOTest.hparams.num_episodes))


if __name__ == '__main__':
  test.main()
