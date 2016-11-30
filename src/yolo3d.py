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
    self.iou_loss = self.batch_iou(self.proposals, self.sensible_zbufs)  


    opt = tf.train.AdamOptimizer(learning_rate=0.0001) #(learning_rate=0.0005)
    
#    opt = tf.train.GradientDescentOptimizer(learning_rate=100)
    self.minimizer = opt.minimize(self.iou_loss)

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
    x = fc(x, 2048, activation='relu') 
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
    proposal_zbufs = tf.map_fn(
      self.proposal2paddedbuf,
      proposals,
      dtype=tf.float32
    )
    min_zbuf = tf.reduce_max(proposal_zbufs, 0)
    flt_zbuf = tf.cast(zbuf, tf.float32)
    sqrdiff = tf.square(tf.sub(min_zbuf, flt_zbuf))
    return tf.reduce_mean(sqrdiff)

  def proposal2paddedbuf(self, proposal):
    px = proposal[0]
    py = proposal[1]
    pz = proposal[2]
    pfw = proposal[3]
    pfh = proposal[4]
    pfz = proposal[5] # ignored
    rot = proposal[6] # ignored
    fx = tf.cast(tf.maximum(tf.minimum(tf.floor(px), self.width), 0.0), tf.int32)
    fy = tf.cast(tf.maximum(tf.minimum(tf.floor(py), self.height), 0.0), tf.int32)
#    fx = tf.cast(tf.floor(tf.sigmoid(px) * self.width), tf.int32)
#    fy = tf.cast(tf.floor(tf.sigmoid(py) * self.height), tf.int32)
#    fx = tf.cast(tf.floor(tf.minimum(tf.maximum(px, 0), 1.0) * self.width), tf.int32)
 #   fy = tf.cast(tf.floor(tf.minimum(tf.maximum(py, 0), 1.0) * self.height), tf.int32)
#    fw = tf.minimum(tf.cast(tf.floor(tf.minimum(tf.maximum(pfw, 0.0), 1.0) * self.width), tf.int32), self.width - fx)
#    fh = tf.minimum(tf.cast(tf.floor(tf.minimum(tf.maximum(pfh, 0.0), 1.0) * self.height), tf.int32), self.height - fy)
    fw = tf.cast(tf.maximum(tf.minimum(pfw, tf.cast(self.width - fx, tf.float32)), 0.0), tf.int32)
    fh = tf.cast(tf.maximum(tf.minimum(pfh, tf.cast(self.height - fy, tf.float32)), 0.0), tf.int32)
    #fz = tf.maximum(tf.minimum(pz * self.depth, self.depth), 0)
    fz = tf.minimum(tf.maximum(pz, 0.0), self.depth - 1)
#    fz = tf.sigmoid(pz) * self.depth
    unpad = tf.fill(
      tf.pack([fw, fh]),
      fz - self.depth + 1
    )
    wpad = tf.pack([fx, self.width  - (fx + fw)])
    hpad = tf.pack([fy, self.height - (fy + fh)])
    padding = tf.pack([wpad, hpad])
    padded = tf.pad(unpad, padding) + (self.depth - 1)
#    padded = tf.Print(tf.pad(unpad, padding), [fx, fy, fw, fh, fz])
#    result = tf.Print(padded, [tf.shape(padded)])
    return padded
    
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
    'batch_size': 40,
    'width': 640,
    'height': 480,
    'depth': 256,
    'channels': 3,
    'pgrid_dims': [10, 8],
    'bb_num': 1,
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

  batch_size, x_in, zbufs = [params[name] for name in ['batch_size', 'x_in', 'zbufs']]
  data_dir = "/data/r9k/obs_data"
#  feed_dict = {params['x_in']: [np.load(data_dir + "/img/4b20069dba.npy")],
#      params['zbufs']: [np.load(data_dir + "/zbuf/4b20069dba.npy")]}
  sess = tf.Session()
  print("built session!")

  tf.scalar_summary('sq_loss', net.iou_loss)
  merged = tf.merge_all_summaries()
  run_id = "r9k-0#bs:{0}".format(batch_size)
  train_writer = tf.train.SummaryWriter("/tmp/{0}".format(run_id), sess.graph)

  fetches = [net.minimizer, net.iou_loss, merged]
  val_fetches = [net.iou_loss, merged]


  sess.run(tf.initialize_all_variables())
  print("done initializing!")
#  import pdb; pdb.set_trace()
  img_loader = obs_data_loader("img")
  zbuf_loader = obs_data_loader("zbuf")
  epoch_size = 2000
  from math import floor
  epochs = 10
  import random
  for e in range(epochs):
    for i in range(0, floor(600 * 1000 * 0.8), batch_size):
      feed_dict = {
          x_in: img_loader[i: i + batch_size],
          zbufs: zbuf_loader[i: i + batch_size]
      }
      trn, loss, summary = sess.run(fetches, feed_dict)
      train_writer.add_summary(summary, i)
      with open("./train_perf", 'a') as trn_f:
        trn_f.write("{0}: {1} : {2}\n".format(e, i, loss))
      print(loss)
      if (floor(i/batch_size) % 200 == 0 and i != 0):
        v_losses = []
        for j, v_i in enumerate(range((600 * 1000) - 2000, 600 * 1000, batch_size)):
          val_feed_dict = {
            x_in: img_loader[i: i + batch_size],
            zbufs: zbuf_loader[i: i + batch_size]
          }
          loss, summary = sess.run(val_fetches, val_feed_dict)
          v_losses.append(loss)
          print(j)
        m_loss = np.mean(v_losses)
        with open("./val_perf", 'a') as trn_f:
          trn_f.write("{0}: {1} : {2}\n".format(e, i, m_loss))
        print("VAL LOSS: {0}".format(m_loss))

  #    print(i * params['batch_size'])

#  import pdb; pdb.set_trace()
#  opt = SGD()
#  opt = Adam()
#  opt = tf.train.AdagradOptimizer(learning_rate = 0.001)
#  import pdb; pdb.set_trace()
#  train_op = TrainOp(net.iou_loss, opt, batch_size = params['batch_size'])
#  trainer = Trainer([train_op], tensorboard_verbose=0)
#  trainer.fit(feed_dicts=[feed_dict], n_epoch=10)
  
