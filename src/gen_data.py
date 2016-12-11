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

object_id_map = {}
missing = set([])

def load_doomitems(path='/usr/lib/python3.5/site-packages/doom_py/src/vizdoom/wadsrc/static/mapinfo/doomitems.txt'):
  di_file = open(path)
  i = 0 
  for line in di_file:
    items = line.strip().split(' ')
    if len(items) == 3:
      object_id_map[items[2]] = i
      i += 1
  object_id_map['Blood'] = i

# setup a game to generate simple data for observation model
def setup_game(scenario = "defend_the_center", show_window=False, show_hud=False, show_weapon=False, show_particles=False):
  game = DoomGame()
  game.load_config("/data/r9k/scenarios/{0}.cfg".format(scenario))
  # need this so it doesn't bug out
  game.add_game_args("+vid_forcesurface 1")
  # mix in the depth buffer and enable other buffers in case we want them
  # This screen format is cv2 friendly, defa/usr/lib/python3.5/site-packages/doom_py/src/vizdoom/wadsrc/static/mapinfo/ult isn't. 
  game.set_screen_format(ScreenFormat.RGB24)
  game.set_depth_buffer_enabled(True)

  game.set_labels_buffer_enabled(True)
  #import pdb; pdb.set_trace()
  #game.clear_available_game_variables()
  #game.add_available_game_variable(GameVariable.POSITION_X)
  #game.add_available_game_variable(GameVariable.POSITION_Y)
  #game.add_available_game_variable(GameVariable.POSITION_Z)
  
  game.set_automap_buffer_enabled(True)
  game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE) 

  game.set_window_visible(show_window)
  game.set_render_hud(show_hud)
  game.set_render_weapon(show_weapon)
  game.set_render_particles(show_particles)

  game.init()

  return game

def save_dat(to_save):
  hash_str = str(hashtime())

  for data_name, data in to_save.iteritems():
    np.save(data_dir + "{0}/{1}".format(data_name, hash_str), data)


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
      automap = state.automap_buffer
      labels_buf = state.labels_buffer
      unique_vals = np.unique(labels_buf)
      rects = np.array([get_bb(labels_buf, value) for value in unique_vals])
      obj_ids = []
      for uv in unique_vals:
        for l in state.labels:
          if l.value == uv:
            name = l.object_name
            obj_ids.append(object_id_map[l.object_name])
            break
      import pdb; pdb.set_trace()
            
      to_save = {
        'img': img,
        'zbuf': zbuf,
        'bbs': rects,
        'automap': automap,
        'obj_ids': obj_ids
      }

      save_dat(to_save)
      game.make_action(random.choice(actions))
      #game.make_action(left)
    print(num_frames)
    sys.stdout.flush()
    num_frames_arr.append(num_frames)
  print missing
  return num_frames_arr
    
def get_bb(buf, value):
  y, x = np.where(buf == value)
  x_min, x_max = np.min(x), np.max(x)
  y_min, y_max = np.min(y), np.max(y)
  return np.array(
    [[x_min, y_min],
     [x_max, y_max]]
  )

def hashtime():
  hash = hashlib.sha1()
  hash.update(str(time.time()) + str(random.randint(1, 100)))
  return hash.hexdigest()[:10]

if __name__ == "__main__":
  game = setup_game()
  load_doomitems()
  num_episodes = int(sys.argv[1].rstrip())
  num_frames = run_and_save_episodes(game, num_episodes)
