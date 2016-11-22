#!/usr/bin/python

import tensorflow as tf
from tflearn.layers.conv import conv_2d as conv2d
from tflearn.layers.conv import max_pool_2d as maxpool
from tflearn.layers.core import fully_connected as fc
from functools import partial

def flatten(l):
  return [item for sublist in l for item in sublist]

def times(n):
  def times_gen(l):
    return flatten([l for i in range(n)])
  return times_gen


class YOLO3D():
  def __init__(self, **params):
    # params: batch_size, width, height, depth, channels,
    #         pgrid_dims, bb_num, num_classes, x_in
    
    # store params as fields
    self.__dict__.update(params)
    
    # generate features
    self.features = self.generate_feature_stack(self.x_in)

    # generate proposals from features
    self.proposal_grid = self.generate_proposals(self.features)
    self.proposals = tf.reshape(
      self.proposal_grid, 
      [self.batch_size, self.num_grid_cells, self.outputs_per_grid_cell]
    )

    # setup coordinate system
    self.coords = gen_coordinate_system(
        self.width, self.height, self.depth
    )
    self.iou_loss = -self.batch_iou(self.proposals, self.zbufs)  

  def generate_feature_stack(self, x_in):
    x = x_in

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

    return x

  def generate_proposals(self, feature_stack):
    x = feature_stack
    x = fc(x, 4096, activation='relu') 
    # proposal = [x, y, z, sigma_x, sigma_y, sigma_z, rot]
    self.outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    self.num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    self.proposal_shape_per_batch =  \
      [self.batch_size] + self.pgrid_dims + [self.outputs_per_grid_cell]

    x = fc(
      x, 
      self.num_grid_cells * self.outputs_per_grid_cell,
      activation='relu'
    )
    x = tf.reshape(x, self.proposal_shape_per_batch)
    return x

  def batch_iou(self, batch_proposals, batch_zbufs):
    per_example_i = tf.map_fn(
      self.per_example_iou, 
      tf.tuple([batch_proposals, batch_zbufs]),
      dtype=tf.float32
    )
    return tf.reduce_mean(per_example_i)

  def per_example_iou(self, arg_tup):
    proposal_sets, zbuf = arg_tup
    proposals = tf.reshape(proposal_sets, [-1, 7])
    func = partial(self.per_proposal_intersect, zbuf=zbuf)
    per_proposal_i = tf.map_fn(func, proposals, dtype=tf.float32)
    return tf.reduce_sum(per_proposal_i)

  def per_proposal_intersect(self, proposal, zbuf):
    proposal_xyz = proposal[0:3]
    proposal_sigma_xyz = proposal[3:6]
    proposal_theta = proposal[6]

    zbuf_as_one_hot = tf.one_hot(
      zbuf,
      self.depth,
      on_value=True,
      off_value=False
    )
    xyz_absolute = tf.boolean_mask(
      self.coords,
      zbuf_as_one_hot
    )
    xyz_relative = xyz_absolute - proposal_xyz

    top = tf.pack([tf.cos(proposal_theta), tf.sin(proposal_theta)])
    bot = tf.pack([-tf.sin(proposal_theta), tf.cos(proposal_theta)])
    rot_mat = tf.pack([top, bot])

    # change to coordinate system about the proposal
    xz_rel = tf.pack(
     [xyz_relative[:, 0],
      xyz_relative[:, 2]],
     axis=0
    )

    # rotate coordinate system with proposal's rotation
    xz_rot = tf.matmul(rot_mat, xz_rel)
    x_rot = xz_rot[0, :]
    z_rot = xz_rot[1, :]
    xyz_rot = tf.pack(
     [x_rot, 
      xyz_relative[:, 1],
      z_rot],
     axis=1
    )
    over_sigma = tf.div(xyz_rot, proposal_sigma_xyz)
    squared = tf.square(over_sigma)
    exponents = -0.5 * tf.reduce_sum(squared, 1)
    f_xyz = tf.exp(exponents)
    
    intersection = tf.reduce_sum(f_xyz)
    return intersection
    
def gen_coordinate_system(width, height, depth):
  x = tf.linspace(-1.0, 1.0, width)
  y = tf.linspace(-1.0, 1.0, height)
  z = tf.linspace(-1.0, 1.0, depth)
  # the indexing param is stupid - otherwise axes swap
  coord = tf.pack(tf.meshgrid(x, y, z, indexing='ij'), axis=3)
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

import tflearn
from tflearn.helpers.trainer import TrainOp, Trainer
from tflearn.optimizers import Adam

from dataloader import obs_data_loader

if __name__ == "__main__":
  params = {
    'batch_size': 1,
    'width': 640,
    'height': 480,
    'depth': 256,
    'channels': 3,
    'pgrid_dims': [7, 7],
    'bb_num': 2,
    'num_classes': 0, # breaks with n_classes > 0 right now
  }

  params['x_in'] = tf.placeholder(
    tf.float32, 
    shape = (params[k] for k in ('batch_size', 'width', 'height', 'channels'))
  )
  params['zbufs'] = tf.placeholder(
    tf.int32,
    shape = (params[k] for k in ('batch_size', 'width', 'height'))
  )

  net = YOLO3D(**params)

  feed_dict = {
    params['x_in']: obs_data_loader("img"), 
    params['zbufs']: obs_data_loader("zbuf")
  }


  train_op = TrainOp(net.iou_loss, Adam().get_tensor(), batch_size = params['batch_size'])
  trainer = Trainer([train_op], tensorboard_verbose=0)
  trainer.fit(feed_dicts=[feed_dict], n_epoch=10)


