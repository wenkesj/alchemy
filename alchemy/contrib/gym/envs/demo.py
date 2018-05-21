# -*- coding: utf-8 -*-
from __future__ import absolute_import

from gym.core import Env
from gym.core import Space
from gym.spaces import discrete

import argparse
import json
import os
import pickle
import string
import time
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import tf_record

from pynput.keyboard import Key
from pynput.keyboard import Listener

from alchemy.utils import assert_utils
from alchemy.contrib.rl import experience
from alchemy.contrib.rl import serialize
from alchemy.multiprocessing import thread_pool


def preload_demos(src, path):
  for fn in os.listdir(path):
    with open(os.path.join(path, fn), 'rb') as f:
      for experiences in pickle.load(f):
        src.write(experiences)


ckeys = set()
str2key = {
  'alt': Key.alt,
  'alt_gr': Key.alt_gr,
  'alt_l': Key.alt_l,
  'alt_r': Key.alt_r,
  'backspace': Key.backspace,
  'caps_lock': Key.caps_lock,
  'cmd': Key.cmd,
  'cmd_l': Key.cmd_l,
  'cmd_r': Key.cmd_r,
  'ctrl': Key.ctrl,
  'ctrl_l': Key.ctrl_l,
  'ctrl_r': Key.ctrl_r,
  'delete': Key.delete,
  'down': Key.down,
  'end': Key.end,
  'enter': Key.enter,
  'esc': Key.esc,
  'f1': Key.f1,
  'home': Key.home,
  'left': Key.left,
  'page_down': Key.page_down,
  'page_up': Key.page_up,
  'right': Key.right,
  'shift': Key.shift,
  'shift_l': Key.shift_l,
  'shift_r': Key.shift_r,
  'space': Key.space,
  'tab': Key.tab,
  'up': Key.up,
}

available_keys_str = 'Available keys: {}'.format(', '.join(list(str2key.keys())))


class EnvInterface(object):
  def __init__(self, keymap_path):
    """Constructs a environment for observations and actions from a KeyBoard.

    Arguments:
      keymap_path: keymap file location.
    """
    self._pressed_keys = set()
    self._keymap = self._parse_keymap(self._read_keymap(keymap_path))

    # We don't want to hog demo frames if threads thrash, so we only dedicate 1 to writing.
    self._demo_queue = thread_pool.ThreadPool(1)
    self._default_action = 0

  def _read_keymap(self, path):
    """Reads a keymap from a `JSON` file.

    Arguments:
      path: keymap file location.

    Returns:
      the keymap as a `dict`.
    """
    with open(path, 'r') as f:
      return json.load(f)

  def _parse_keymap(self, obj):
    assert_utils.assert_true(
        'default' in obj,
        '`keymap` must have a `default` field for missing or idle keys/actions')

    self._default_action = int(obj['default'])

    keymap = dict()
    for key, val in obj.items():
      keys = list()
      key = key.translate(str.maketrans('', '', string.whitespace))
      keystrs = key.split('+')
      for ks in keystrs:
        if ks not in str2key:
          raise KeyError(
              '`key` must be a valid key, found {}.'.format(
                  ks) + available_keys_str)
        keys.append(str2key[ks])
      keymap[frozenset(keys)] = int(val)
    return keymap

  def _write_demonstrations(self,
                            demonstrations,
                            save_dir,
                            state_dtype,
                            action_dtype,
                            action_value_dtype,
                            reward_dtype):
    record_file_path = os.path.join(save_dir, str(int(time.time())))
    with tf_record.TFRecordWriter(record_file_path) as writer:
      for experiences in demonstrations:
        replay = experience.Replay(*zip(*experiences))
        serialized_replay = serialize.serialize_replay(
            replay,
            state_dtype,
            action_dtype,
            action_value_dtype,
            reward_dtype)
        writer.write(serialized_replay.SerializeToString())

  def record(self, env, save_dir,
             max_sequence_length=200,
             min_sequence_length=10,
             max_demo_length=10,
             min_demo_length=1,
             max_episodes=-1,
             frame_sleep_time=0.056):
    """Allows one to control and record a discrete space `gym.Env` using the keyboard.

    Arguments:
      env: a `gym.Env` instance. Must have a categorical `action_space`.
      save_dir: `str`, the directory to write recordings.
      max_sequence_length: `int`, max frames to play until dumping the demo and starting a new one.
          Note: This doesn't stop the reset environment.
      min_sequence_length: `int`, min frames to play until dumping the demo.
      max_demo_length: `int`, max number of demos to dump. This stops saving and ultimately stops
          the environment.
      min_demo_length: 'int', the minumum number of demos to dump.
      max_episodes: 'int', the max number of possible episodes to record from. This is the last
          exit case.
    """
    assert_utils.assert_true(
        isinstance(env, Env),
        '`env` must be an instance of `gym.Env`')
    assert_utils.assert_true(
        env.action_space is not None,
        '`env.action_space` property must be set.')
    assert_utils.assert_true(
        isinstance(env.action_space, Space),
        '`env.action_space` must be an instance of `gym.Space`')
    assert_utils.assert_true(
        env.observation_space is not None,
        '`env.observation_space` property must be set.')
    assert_utils.assert_true(
        isinstance(env.observation_space, Space),
        '`env.observation_space` must be an instance of `gym.Space`')

    # TODO(wenkesj): remove this when a Space-to-Distribution is implemented
    assert_utils.assert_true(
        isinstance(env.action_space, discrete.Discrete),
        ','.join(['IM SORRY: `env.action_space` must be an instance of `gym.spaces.Discrete`',
                  'check back later']))

    global ckeys
    if save_path:
      try: os.makedirs(save_path)
      except: pass

    state_dtype = type_utils.safe_tf_dtype(env.observation_space.dtype)
    action_dtype = type_utils.safe_tf_dtype(env.action_space.dtype)
    action_value_dtype = dtypes.float32
    reward_dtype = dtypes.float32

    keys_to_action = self._keymap
    action_to_action_values = np.identity(env.action_space.n)
    next_state = env.reset()
    env.render()

    demonstrations = []
    experiences = []

    def on_press(key):
      global ckeys
      ckeys.add(key)

    def on_release(key):
      global ckeys
      ckeys.remove(key)
      if key == Key.esc:
        return False

    stoppable = max_episodes > 0
    try:
      action = 0
      episode = 0
      with Listener(on_press=on_press, on_release=on_release) as listener:
        while True:
          if stoppable:
            if episode >= max_episodes:
              break
          time.sleep(fps)

          action = keys_to_action.get(frozenset(ckeys), default_action)
          state = next_state
          next_state, reward, terminal, _ = env.step(action)
          action_values = action_to_action_values[action]

          experiences.append(
              experience.Experience(
                  state, next_state,
                  action, action_values,
                  reward, terminal))

          if len(experiences) >= max_sequence_length:
            demonstrations.append(experiences)
            experiences = []

          if save_path:
            if len(demonstrations) >= max_demo_length:
              self.demo_queue.add_task(
                  self._write_demonstrations,
                  demonstrations,
                  save_dir,
                  state_dtype,
                  action_dtype,
                  action_value_dtype,
                  reward_dtype)
              demonstrations = []

          if terminal:
            if len(experiences) > min_sequence_length:
              demonstrations.append(experiences)
            else:
              experiences = []
            next_state = env.reset()
            episode += 1

    except (KeyboardInterrupt, EOFError):
      listener.join()
      pass

    if len(experiences) > min_sequence_length:
      demonstrations.append(experiences)
    if len(demonstrations) > min_demo_length:
      self.demo_queue.add_task(
          self._write_demonstrations,
          demonstrations,
          save_dir,
          state_dtype,
          action_dtype,
          action_value_dtype,
          reward_dtype)
    self.demo_queue.wait_completion()
