#!/usr/bin/python2

from vizdoom import *
import random
import time
import hashlib
import numpy as np
import sys
import math

# action definition
left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

data_dir = "/data/r9k/obs_data/"

object_id_map = {
    'MarineChainsaw': 0,
    'Demon': 1,
    'Blood': 2
}

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

def find_cell(x, y):
  return math.floor(x * y/2) / (y/2)

def find_idx(x, y):
  return x * (y/2) + (y/2)

def run_and_save_episodes(game, num_episodes):
  seen = set([])
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

      rects = []
      obj_ids = []
      zs = []
      for uv in unique_vals:
        for l in state.labels:
          if l.value == uv:
            name = l.object_name
            seen.add(name)
            rects.append(get_bb(labels_buf, uv))
            zs.append(np.mean(zbuf[labels_buf == uv]))
            obj_ids.append(object_id_map[l.object_name])
            break
      

      grid = np.zeros((10, 8), dtype=np.dtype('O'))
      for j in range(len(rects)):
          rect = rects[j]
          z = zs[j]
          obj_id = obj_ids[j]
          
          xmin = rect[0, 0] / 320.0 - 1
          xmax = rect[1, 0] / 320.0 - 1
          xcenter = (xmin + xmax) / 2
          ymin = rect[0, 1] / 240.0 - 1
          ymax = rect[1, 1] / 240.0 - 1
          ycenter = (ymin + ymax) / 2

          cell_x_min = find_cell(xcenter, 10)
          cell_y_min = find_cell(ycenter, 8)
          # care about center relative to box min
          xcenter -= cell_x_min
          ycenter -=cell_y_min

          xidx = find_idx(cell_x_min, 10)
          yidx = find_idx(cell_y_min, 8)
          
          coords = np.array([xcenter, ycenter, math.sqrt(xmax - xmin), math.sqrt(ymax- ymin), z/ 60 - 1])
          import pdb; pdb.set_trace()
          cur = grid[xidx, yidx]
          if cur == 0:
              classes = np.zeros((3,))
              classes[obj_id] = 1
              grid[xidx, yidx] = (coords.reshape((1,5)), classes)
          else:
              cur[1][obj_id] += 1 
              grid[xidx, yidx] = (np.vstack(cur[0], coords), cur[1])

      for j in range(10):
          for k in range(8):
              if grid[j, k] == 0:
                  grid[j, k] = (None, np.zeros((3,)))
          
      
     


      
      #import pdb; pdb.set_trace()
            
      to_save = {
        'img': img,
        'zbuf': zbuf,
        'bbs': rects,
        'automap': automap,
        'obj_ids': obj_ids
      }

      #save_dat(to_save)
      game.make_action(random.choice(actions))
      #game.make_action(left)
    #print(num_frames)
    sys.stdout.flush()
    num_frames_arr.append(num_frames)
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
  num_episodes = int(sys.argv[1].rstrip())
  num_frames = run_and_save_episodes(game, num_episodes)
