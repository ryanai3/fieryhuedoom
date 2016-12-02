import scipy.misc
import numpy as np

def npy_to_img(fname):
  npy = np.load('/data/r9k/obs_data/automap/' + fname + '.npy')
  npy = np.where(npy>200, 1, 0)
  scipy.misc.imsave('/data/r9k/obs_data/automap_img/' + fname +'.jpg', npy[:,:,0])

def batch_convert(list_fname):
  f = open(list_fname)
  for line in f:  
    npy_to_img(line.split('.')[0])
