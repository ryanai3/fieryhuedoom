#!/usr/bin/python

import tensorflow as tf
from tflearn.layers.conv import conv_2d as conv2d
from tflearn.layers.conv import max_pool_2d as maxpool
from tflearn.layers.core import fully_connected as fc

def flatten(l):
  return [item for sublist in l for item in sublist]

def times(n):
  def times_gen(l):
    return flatten([l for i in range(n)])
  return times_gen


class YOLO3D():
  def __init__(self, batch_size, width, height, channels, pgrid_size, bb_num,
      num_classes, x_in):

    x = tf.placeholder(tf.float32, shape=(batch_size, width, height, channels))
    x = conv_then_pool_block({
      'convs'    : [(7, 7, 64, 2)],
      'max_pool' : [(2, 2, 2)]
    })(x)

    x = conv_then_pool_block({
      'convs'    : [(3, 3, 192)],
      'max_pool' : [(2, 2, 2)]
    })(x)

    x = conv_then_pool_block({
      'convs': [
        (1, 1, 128),
        (3, 3, 256),
        (1, 1, 256),
        (3, 3, 512)
      ],
      'max_pool' : [(2, 2, 2)]
    })(x)

    x = conv_then_pool_block({
      'convs': 
        times(4)([
          (1, 1, 256), 
          (3, 3, 512)
        ]) + [
        (1, 1, 512),
        (3, 3, 1024)
        ],
      'max_pool' : [(2, 2, 2)]
    })(x)

    x = conv_block(
      times(2)([
        (1, 1, 512), 
        (3, 3, 1024)
      ]) + [
        (3, 3, 1024), 
        (3, 3, 1024, 2)
      ]
    )(x)

    x = conv_block(
      times(2)([(3, 3, 1024)])
    )(x)

    x = fc(x, 4096, activation='relu') 
    # bb_num * 6 == x, y, z, sigma_x, sigma_y, sigma_z, rot
    outputs_per_grid_cell = ((bb_num * 7) + num_classes))
    x = fc(x, (pgrid_size ** 2) * outputs_per_grid_cell, activation='relu')
    x = tf.reshape(t, [7, 7, outputs_per_grid_cell])
    
    import pdb; pdb.set_trace()

  def gen_coordinate_system(self, width, height, depth):
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    z = tf.linspace(-1.0, 1.0, depth)
    coord = tf.meshgrid(x, y, z)
    return coord
    

def conv_block(convs_params):
  def gen_conv_block(x):
    for cp in convs_params:
      if len(cp) == 3:
        x = conv2d(x, cp[2], [cp[0], cp[1]], 
          padding='same', activation='relu'
        )
      elif len(cp) == 4:
        x = conv2d(x, cp[2], [cp[0], cp[1]], [cp[3], cp[3]],
          padding='same', activation='relu'
        )
      else:
        print("didn't understand conv block params")
    return x
  return gen_conv_block

def conv_then_pool_block(param_dict):
  convs_params = param_dict['convs']
  def gen_ctp_block(x):
    x = conv_block(convs_params)(x)
    try:
      mp = param_dict['max_pool']
      x = max_pool_2d(x, [mp[0], mp[1]], [mp[2], mp[2]], padding='same')
    except:
      pass
    return x
  return gen_ctp_block

if __name__ == "__main__":
  net = YOLO3D(4, 640, 480, 3, None)
  co = net.gen_coordinate_system()
  import pdb; pdb.set_trace()
  print(32)
  print(32)
  

