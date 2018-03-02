# -*- coding: utf-8 -*-
import argparse, json, os, pickle, string, time

import numpy as np

from pynput.keyboard import Key, Listener

from alchemy.memory import Trajectory, Transition


def preload_demos(src, path):
  for fn in os.listdir(path):
    with open(os.path.join(path, fn), 'rb') as f:
      for traj in pickle.load(f):
        src.write(traj)


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


class DemoRecorder(object):
  def __init__(self, keymap_path):
    """Constructs a demonstration record environment for observations and actions of humans.

    Args:
      path: keymap file location.
    """
    self._pressed_keys = set()
    self._keymap = self._parse_keymap(self._read_keymap(keymap_path))

  def _read_keymap(self, path):
    """Reads a keymap from a `JSON` file.

    Args:
      path: keymap file location.
    """
    with open(path, 'r') as f:
      return json.load(f)

  def _parse_keymap(self, obj):
    keymap = dict()
    for key, val in obj.items():
      keys = list()
      key = key.translate(str.maketrans('', '', string.whitespace))
      keystrs = key.split('+')
      for ks in keystrs:
        k = str2key.get(ks, '0')
        keys.append(k)
      keymap[frozenset(keys)] = int(val)
    return keymap

  def _save_demo(self, save_path, demonstrations, episode):
    with open(os.path.join(save_path, 'episode_{}'.format(episode)), 'wb') as f:
      pickle.dump(demonstrations, f, protocol=pickle.HIGHEST_PROTOCOL)

  def play(self, env, num_actions,
           save_path=None,
           demo_ptr=0,
           max_sequence_length=200,
           min_sequence_length=10,
           max_demo_length=10,
           min_demo_length=1,
           max_episodes=-1):
    """Allows one to play the game using keyboard."""
    global ckeys

    if save_path:
      try: os.makedirs(save_path)
      except: pass
    keys_to_action = self._keymap

    values = np.identity(num_actions)

    next_state = env.reset()
    env.render()

    demonstrations = []
    traj = Trajectory([])

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
          time.sleep(0.056)

          action = keys_to_action.get(frozenset(ckeys), 0)
          state = next_state
          next_state, reward, terminal, info = env.step(action)

          traj.add(
              Transition(state=state,
                         action=action,
                         values=values[action],
                         reward=reward,
                         terminal=terminal,
                         info=info))

          if len(traj) >= max_sequence_length:
            demonstrations.append(traj)
            traj = Trajectory([])

          if save_path:
            if len(demonstrations) >= max_demo_length:
              self._save_demo(save_path, demonstrations, demo_ptr)
              demo_ptr += 1
              demonstrations = []

          if terminal:
            if len(traj) > min_sequence_length:
              demonstrations.append(traj)
            else:
              traj = Trajectory([])
            next_state = env.reset()
            episode += 1

    except (KeyboardInterrupt, EOFError):
      listener.join()
      pass
    if len(traj) > min_sequence_length:
      demonstrations.append(traj)
    return demo_ptr, demonstrations

  def record(self, env, num_actions, save_path=None,
             demo_ptr=0,
             max_sequence_length=200,
             min_sequence_length=10,
             max_demo_length=10,
             min_demo_length=1,
             max_episodes=-1):
    """Records a set of demonstrations from the given `env` and stores them to directory `path`.

    Args:
      env: environment to record demonstrations.
      path: path of the directory to store the demonstrations.
    """
    demo_ptr, demonstrations = self.play(
        env, num_actions,
        save_path=save_path,
        demo_ptr=demo_ptr,
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        max_demo_length=max_demo_length,
        min_demo_length=min_demo_length,
        max_episodes=max_episodes)

    if save_path:
      if len(demonstrations) > min_demo_length:
        self._save_demo(save_path, demonstrations, demo_ptr)
    return demonstrations

  @staticmethod
  def get_args(parser=None):
    if parser is None:
      parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--key_map', help='json key map location', type=str, required=True)
    parser.add_argument('--save_path', help='demo save path', type=str, default='')
    parser.add_argument('--start_demo_ptr', help='demo start file idx', type=int, default=0)
    parser.add_argument('--max_sequence_length', help='max sequence length',
                        type=int, default=200)
    parser.add_argument('--min_sequence_length', help='min sequence length',
                        type=int, default=10)
    parser.add_argument('--max_demo_length', help='max demo length', type=int, default=10)
    parser.add_argument('--min_demo_length', help='min demo length', type=int, default=1)
    parser.add_argument('--max_episodes', help='number of episodes to record', type=int, default=1)
    parser.add_argument('--num_actions', help='number of actions', type=int, required=True)
    return parser

  @staticmethod
  def main(env, parser=None):
    if parser is None:
      parser = DemoRecorder.get_args()
    args = parser.parse_args()
    demo = DemoRecorder(keymap_path=args.key_map)

    return demo.record(
        env, args.num_actions,
        save_path=args.save_path,
        demo_ptr=args.start_demo_ptr,
        max_sequence_length=args.max_sequence_length,
        min_sequence_length=args.min_sequence_length,
        max_demo_length=args.max_demo_length,
        min_demo_length=args.min_demo_length,
        max_episodes=args.max_episodes)
