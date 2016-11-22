#!/usr/bin/python
import numpy as np

r9k_data = "/data/r9k/obs_data/"


# tflearn release doesn't have this :(
# why aren't they *BLEEDING EDGE* ?!?! /s
class Preloader(object):
  def __init__(self, array, function):
    self.array = array
    self.function = function

  def __getitem__(self, id):
    if type(id) in [list, np.ndarray]:
      return [self.function(self.array[i]) for i in id]
    elif isinstance(id, slice):
      return [self.function(arr) for arr in self.array[id]]
    else:
      return self.function(self.array[id])

  def __len__(self):
    return len(self.array)

class NpyLoader(Preloader):
  def __init__(self, path_arr):
    fn = lambda x: self.preload(x)
    super(NpyLoader, self).__init__(path_arr, fn)

  def preload(self, path):
    return np.load(path)

def obs_data_loader(mode):
  data_dir = r9k_data + "{0}/".format(mode)
  filelist_path = r9k_data + "files.list"
  filenames = [data_dir + line.rstrip() 
    for line in open(filelist_path, 'r').readlines()
  ]

  return NpyLoader(filenames) 
