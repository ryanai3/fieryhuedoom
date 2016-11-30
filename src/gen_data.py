#!/usr/bin/python2

from vizdoom import *
import random
import time
import hashlib
import numpy as np
import sys

# action definition
left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

data_dir = "/data/r9k/obs_data/"

# setup a game to generate simple data for observation model
def setup_game(scenario = "defend_the_center"):
  game = DoomGame()
  game.load_config("/data/r9k/scenarios/{0}.cfg".format(scenario))
  # need this so it doesn't bug out
  game.add_game_args("+vid_forcesurface 1")
  # mix in the depth buffer and enable other buffers in case we want them
  # This screen format is cv2 friendly, default isn't. 
  game.set_screen_format(ScreenFormat.RGB24)
  game.set_depth_buffer_enabled(True)
  game.set_labels_buffer_enabled(True)
  game.set_automap_buffer_enabled(True)
  game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE) 

  # so we can run this without an x server
  game.set_window_visible(False)
  game.set_render_hud(False)
  game.set_render_weapon(False)

  # so we can run this without an x server
  # game.set_window_visible(False)
  game.init()

  return game

def save_dat(img, zbuf, bbs):
  hash_str = str(hashtime())
  img_fname = data_dir + "img/{0}".format(hash_str)
  zbuf_fname = data_dir + "zbuf/{0}".format(hash_str)
  modes = ['img', 'zbuf', 'bbs']
  img_fname, zbuf_fname, bbs_fname =  \
    [data_dir + "{0}/{1}".format(mode, hash_str) for mode in modes]
  np.save(img_fname, img)
  np.save(zbuf_fname, zbuf) 
  np.save(bbs_fname, bbs)

def run_and_save_episodes(game, num_episodes):
  num_frames_arr = []
  for i in range(num_episodes):
    game.new_episode()
    num_frames = 0
    while not game.is_episode_finished():
      num_frames += 1
      state = game.get_state()
      img = state.screen_buffer
      zbuf = state.depth_buffer
      labels = state.labels_buffer
      rects = np.array([get_bb(labels, value) for value in np.unique(labels)])
      save_dat(img, zbuf, rects) 
#      class_map = {} # TODO: make this from state.labels
      game.make_action(random.choice(actions))
    print(num_frames)
    sys.stdout.flush()
    num_frames_arr.append(num_frames)
  return num_frames_arr
    
def get_bb(buf, value):
  y, x = np.where(buf == value)
  x_min, x_max = np.min(x), np.max(x)
  y_min, y_max = np.min(y), np.min(y)
  return np.array(
    [[x_min, y_min],
     [y_min, y_max]]
  )

def hashtime():
  hash = hashlib.sha1()
  hash.update(str(time.time()) + str(random.randint(1, 100)))
  return hash.hexdigest()[:10]

if __name__ == "__main__":
  game = setup_game()
  num_episodes = int(sys.argv[1].rstrip())
  num_frames = run_and_save_episodes(game, num_episodes)
