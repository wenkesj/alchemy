# -*- coding: utf-8 -*-
from __future__ import absolute_import

import gym
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.training import training_util
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import adam

from alchemy.contrib.rl import dataset
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import streams
from alchemy.contrib.rl.pg import pg_ops


def mlp(inputs, hidden_layers, num_actions):
  hidden = inputs
  for hidden_size in hidden_layers:
    hidden = core.dense(
        hidden, units=hidden_size, use_bias=False, activation=nn_ops.relu)
  logits = core.dense(
      hidden, units=num_actions, use_bias=False)
  return logits


def mlp_policy(action_values, exploration, deterministic, action_dtype):
  shape = array_ops.shape(action_values)
  deterministic_action = lambda: math_ops.argmax(
      action_values, axis=-1, output_type=action_dtype)
  return control_flow_ops.cond(
      deterministic,
      deterministic_action,
      lambda: control_flow_ops.cond(
          exploration < random_ops.random_uniform([]),
          deterministic_action,
          lambda: math_ops.argmax(
              random_ops.random_uniform(shape),
              axis=-1, output_type=action_dtype)))


class PGTest(test.TestCase):

  hparams = hparam.HParams(
      learning_rate=1e-3,
      hidden_layers=[8],
      initial_exploration=.5,
      discount=.99,
      exploration_decay_steps=4*5,
      exploration_decay_rate=.99,
      max_sequence_length=200,
      num_episodes=64,
      batch_size=16,
      num_epochs=2,
      num_iterations=10)

  def test_pg_ops_advantage(self):
    np.random.seed(42)
    random_seed.set_random_seed(42)

    env = gym.make('CartPole-v0')
    action_value_shape = [2]
    action_value_dtype = dtypes.float32
    stream = streams.SimpleReplayStream.from_gym_env(
        env, action_value_shape, action_value_dtype)

    global_step = training_util.get_or_create_global_step()
    state_ph = array_ops.placeholder(
        stream.state_dtype,
        [None, None] + list(stream.state_shape),
        name='state')
    action_ph = array_ops.placeholder(
        stream.action_dtype,
        [None, None] + list(stream.action_shape),
        name='action')
    advantage_ph = array_ops.placeholder(
        stream.reward_dtype,
        [None, None],
        name='advantage')
    sequence_length_ph = array_ops.placeholder(
        dtypes.int32,
        [None, 1],
        name='sequence_length')
    deterministic_ph = array_ops.placeholder(
        dtypes.bool,
        [],
        name='deterministic')

    replay_dataset = dataset.ReplayDataset(
        stream, max_sequence_length=PGTest.hparams.max_sequence_length)
    replay_dataset = replay_dataset.batch(PGTest.hparams.batch_size)
    replay_dataset = replay_dataset.repeat(PGTest.hparams.num_epochs)
    replay_op = replay_dataset.make_one_shot_iterator().get_next()

    sequence_length = array_ops.squeeze(replay_op.sequence_length, -1)
    advantage_op = pg_ops.advantage(
        replay_op.reward,
        sequence_length=sequence_length,
        max_sequence_length=PGTest.hparams.max_sequence_length,
        discount=PGTest.hparams.discount)

    exploration_op = learning_rate_decay.exponential_decay(
        PGTest.hparams.initial_exploration,
        global_step,
        PGTest.hparams.exploration_decay_steps,
        PGTest.hparams.exploration_decay_rate)

    action_values_op = mlp(state_ph, PGTest.hparams.hidden_layers, action_value_shape[-1])
    action_op = mlp_policy(action_values_op, exploration_op, deterministic_ph, stream.action_dtype)

    loss_op = losses_impl.sparse_softmax_cross_entropy(
        action_ph, action_values_op, reduction=losses_impl.Reduction.NONE)
    loss_op *= advantage_ph
    loss_op = math_ops.reduce_mean(
        math_ops.reduce_sum(loss_op, axis=-1) / math_ops.cast(
            array_ops.squeeze(sequence_length_ph, -1), loss_op.dtype))

    optimizer = adam.AdamOptimizer(
        learning_rate=PGTest.hparams.learning_rate)
    train_op = optimizer.minimize(loss_op)

    def run_eval(sess, deterministic=False, save_replay=True):
      rewards = 0.
      for episode in range(PGTest.hparams.num_episodes):
        experiences = []
        next_state = env.reset()

        while True:
          state = next_state
          action_values, action = sess.run(
              (action_values_op, action_op),
              feed_dict={
                  state_ph: [[state]],
                  deterministic_ph: deterministic
              })
          next_state, reward, terminal, _ = env.step(np.squeeze(action))
          experiences.append(experience.Experience(
              state, next_state, action, action_values, reward, terminal))
          if terminal:
            break
        replay = experience.Replay(
            *zip(*experiences), sequence_length=len(experiences))
        if save_replay:
          stream.write(replay)
        rewards += sum(replay.reward)
      return rewards

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      for iteration in range(PGTest.hparams.num_iterations):
        rewards = run_eval(sess, deterministic=False, save_replay=True)
        while True:
          try:
            replay, advantage = sess.run((replay_op, advantage_op))
          except (errors_impl.OutOfRangeError, errors_impl.InvalidArgumentError):
            break
          _, loss = sess.run(
              (train_op, loss_op),
              feed_dict={
                  state_ph: replay.state,
                  action_ph: replay.action,
                  advantage_ph: advantage,
                  sequence_length_ph: replay.sequence_length,
                  deterministic_ph: True,
              })
        rewards = run_eval(sess, deterministic=True, save_replay=False)
        print('average_rewards = {}'.format(rewards / PGTest.hparams.num_episodes))


if __name__ == '__main__':
  test.main()
