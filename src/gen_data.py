#!/usr/bin/python2

from vizdoom import *


# action definition
left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

# setup a game to generate simple data for observation model
def setup_game(scenario = "defend_the_center"):
  game = DoomGame()
  game.load_config("/data/r9k/scenarios/{0}.cfg".format(scenario))
  # need this so it doesn't bug out
  game.add_game_args("+vid_forcesurface 1")
  # mix in the depth buffer
  game.set_screen_format(ScreenFormat.CRCGCBDB)
  game.init()
  return game

if __name__ == "__main__":
  print(32)
  import pdb; pdb.set_trace()
  print(32)
  print(32)


