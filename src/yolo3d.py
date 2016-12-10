#!/usr/bin/python

import tensorflow as tf
from tflearn.layers.conv import conv_2d as conv2d
from tflearn.layers.conv import max_pool_2d as maxpool
from tflearn.layers.core import fully_connected as fc
from tflearn.layers.core import dropout
from functools import partial
from keras.layers.core import RepeatVector

from math import floor

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

    self.upleft, self.botright = [
      tf.placeholder(tf.float32, [self.batch_size] + self.pgrid_dims + [self.bb_num, 2])
      for i in range(2)
    ]
    self.areas = tf.placeholder(tf.float32, [self.batch_size] + self.pgrid_dims + [self.bb_num])

    self.features = self.generate_feature_stack(self.sensible_x_in)

    # generate proposals from features
    self.proposal_grid = self.generate_proposals(self.features)
    self.proposals = tf.reshape(
      self.proposal_grid, 
      [self.batch_size, self.num_grid_cells, self.outputs_per_grid_cell]
    )
    opt = tf.train.AdamOptimizer(learning_rate=0.0001) #(learning_rate=0.0005)
    self.loss = self.gen_loss()
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
    x = fc(x, 2048, activation='leaky_relu') 
    # proposal = [x, y, z, sigma_x, sigma_y, sigma_z, rot]
    self.outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    self.num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    self.proposal_shape_per_batch =  \
      [self.batch_size] + self.pgrid_dims + [self.outputs_per_grid_cell]
    x = dropout(x, self.dropout_prob)
    x = fc(
      x, 
      self.num_grid_cells * self.outputs_per_grid_cell,
      activation='linear' 
    )
    x = tf.reshape(x, self.proposal_shape_per_batch)
    return x

  def gen_loss(self):
    import pdb; pdb.set_trace()
    coords = tf.reshape(
      self.proposal_grid[:, :, :, (self.num_classes + self.bb_num):],
      [self.batch_size] + self.pgrid_dims + [self.bb_num, -1]
   )
    cell_width = floor(self.width / self.pgrid_dims[0])
    cell_height = floor(self.height / self.pgrid_dims[1])
    wh = tf.pow(coords[:, :, :, :, 2:4], 2) 
    area_pred = wh[:, :, :, :, 0] * wh[:, :, :, :, 1]
    centers = coords[:, :, :, :, 0:2]
    bb_floor = centers - (wh * 0.5)
    bb_ceil = centers + (wh * 0.5)

    # calculate intersection areas:
    intersect_upleft   = tf.maximum(bb_floor, self.upleft)
    intersect_botright = tf.minimum(bb_ceil, self.botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.mul(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    # calculate the best IOU, set "responsibility" for predictions
    iou = tf.div(intersect, self.areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.mul(best_box, self.confs)

    # take care of weights in loss
    weight_con = self.snoob * (1.0 - confs) + self.sconf * confs
    conid = tf.mul(_conid, weight_con)
    weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
    cooid = tf.mul(_cooid, scoor * weight_coo)
    proid = sprob * self.proid

    import pdb; pdb.set_trace()

def conv_block(convs_params):
  def gen_conv_block(x):
    for cp in convs_params:
      if len(cp) == 3:
        x = conv2d(x, cp[2], [cp[0], cp[1]], 
          padding='same', activation='leaky_relu'
        )
      elif len(cp) == 4:
        x = conv2d(x, cp[2], [cp[0], cp[1]], [cp[3], cp[3]],
          padding='same', activation='leaky_relu'
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
    'dropout_prob': 0.5,
    'width': 640,
    'height': 480,
    'depth': 256,
    'channels': 3,
    'pgrid_dims': [10, 8],
    'bb_num': 2,
    'num_classes': 1,
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
  img_loader = obs_data_loader("img")
  zbuf_loader = obs_data_loader("zbuf")
  epoch_size = 2000
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
  
