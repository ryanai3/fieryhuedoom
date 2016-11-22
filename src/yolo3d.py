#!/usr/bin/python

import tensorflow as tf
from tflearn.layers.conv import conv_2d as conv2d
from tflearn.layers.conv import max_pool_2d as maxpool
from tflearn.layers.core import fully_connected as fc
from functools import partial
from keras.layers.core import RepeatVector

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
    self.sensible_x_in = tf.transpose(self.x_in, [0, 3, 2, 1])
    self.sensible_x_in = tf.cast(self.sensible_x_in, tf.float32)

    self.sensible_zbufs = tf.transpose(self.zbufs, [0, 2, 1])
    self.flt_zbufs = tf.cast(self.sensible_zbufs, tf.float32)
    self.sensible_zbufs = tf.cast(self.sensible_zbufs, tf.int32)


    self.features = self.generate_feature_stack(self.sensible_x_in)

    # generate proposals from features
    self.proposal_grid = self.generate_proposals(self.features)
    self.proposals = tf.reshape(
      self.proposal_grid, 
      [self.batch_size, self.num_grid_cells, self.outputs_per_grid_cell]
    )

    # setup coordinate system
    with tf.variable_scope("topline"):
      self.coords = gen_coordinate_system(
          self.width, self.height, self.depth
      )
    self.iou_loss = -self.batch_iou(self.proposals, self.sensible_zbufs)  

#    self.err = tf.reduce_sum(tf.square(self.pred_z - self.flt_zbufs))
#    opt = tf.train.AdagradOptimizer(learning_rate=0.001)
#    self.minimizer = opt.minimize(self.err)

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

    x = conv_block([(3, 3, 1024)])(x)
    x = conv_block([(3, 3, 512)])(x)

#    x = conv_block(
#      times(2)([(3, 3, 1024)])
#    )(x)

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
    zbuf_r_n = self.num_grid_cells * self.bb_num
    zbuf_r = tf.transpose(RepeatVector(zbuf_r_n)(zbuf), [1, 0, 2])
    proposals = tf.reshape(proposal_sets, [-1, 7])
#    func = partial(self.per_proposal_intersect, zbuf=zbuf)
    per_proposal_i = tf.map_fn(
      self.per_proposal_intersect,
      tf.tuple([proposals, zbuf_r]),
      dtype=tf.float32
    )
    return tf.reduce_mean(per_proposal_i)

  def per_proposal_intersect(self, arg_tup):
    proposal, zbuf = arg_tup
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
#      coords,
      zbuf_as_one_hot
    )
    xyz_relative = xyz_absolute - proposal_xyz

    top = tf.pack([tf.cos(proposal_theta), -tf.sin(proposal_theta)])
    bot = tf.pack([tf.sin(proposal_theta), tf.cos(proposal_theta)])
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
    over_sigma = tf.div(xyz_rot, proposal_sigma_xyz + tf.constant(1e-7))
    squared = tf.square(over_sigma)
    exponents = tf.mul(-0.5, tf.reduce_sum(squared, 1))
    f_xyz = tf.exp(exponents)
    intersection = tf.Print(tf.reduce_sum(f_xyz), [s])
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
    for mp in param_dict['max_pool']:
      x = maxpool(x, [mp[0], mp[1]], [mp[2], mp[2]], padding='same')
    return x
  return gen_ctp_block

import tflearn
from tflearn.helpers.trainer import TrainOp, Trainer
#from tflearn.optimizers import Adam, SGD, AdaGrad

from dataloader import obs_data_loader

if __name__ == "__main__":
  params = {
    'batch_size': 1,
    'width': 640,
    'height': 480,
    'depth': 256,
    'channels': 3,
    'pgrid_dims': [10, 8],
    'bb_num': 2,
    'num_classes': 0, # breaks with n_classes > 0 right now
  }

  params['x_in'] = tf.placeholder(
    tf.uint8, 
    shape = (params[k] for k in ('batch_size', 'channels', 'height', 'width'))
  )

  params['zbufs'] = tf.placeholder(
    tf.uint8,
    shape = (params[k] for k in ('batch_size', 'height', 'width'))
  )

  net = YOLO3D(**params)

  def ret_feed_dict(i):
    return {
        params['x_in']: obs_data_loader("img")[i: i + 1],
        params['zbufs']: obs_data_loader("zbuf")[i: i + 1]
    }
  feed_dict = {
      params['x_in']: obs_data_loader("img")[0:params['batch_size']],
      params['zbufs']: obs_data_loader("zbuf")[0:params['batch_size']]
  }

  import numpy as np

  data_dir = "/data/r9k/obs_data"
#  feed_dict = {params['x_in']: [np.load(data_dir + "/img/4b20069dba.npy")],
#      params['zbufs']: [np.load(data_dir + "/zbuf/4b20069dba.npy")]}
  fetches = [net.iou_loss]
  sess = tf.Session()
  print("built session!")
  sess.run(tf.initialize_all_variables())
  print("done initializing!")
#  import pdb; pdb.set_trace()
  for i in range(100):
    proposals = sess.run(fetches, ret_feed_dict(i))
    import pdb; pdb.set_trace()
    print(i * params['batch_size'])

#  import pdb; pdb.set_trace()
#  opt = SGD()
#  opt = Adam()
#  opt = tf.train.AdagradOptimizer(learning_rate = 0.001)
#  import pdb; pdb.set_trace()
#  train_op = TrainOp(net.iou_loss, opt, batch_size = params['batch_size'])
#  trainer = Trainer([train_op], tensorboard_verbose=0)
#  trainer.fit(feed_dicts=[feed_dict], n_epoch=10)
  
