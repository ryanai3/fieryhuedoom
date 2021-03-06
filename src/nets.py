#!/usr/bin/python2

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable

def listify(x):
  l = x
  if type(x) != list:
    l = [x]
  for x_i in l:
    yield x_i

def flatten(l):
  return[item for sublist in l for item in sublist]

def repeat(n):
  def repeat_gen(l):
    return flatten([l for i in range(n)])
  return repeat_gen

class ControlYOLO(chainer.Chain):
  def __init__(self, **params):
    # params: pgrid_dims, bb_num, num_classes

    # store params as fields
    self.__dict__.update(params)
    
    outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    vec_len = num_grid_cells * outputs_per_grid_cell
    self.proposal_grid_shape = self.pgrid_dims + [outputs_per_grid_cell]

    super(ControlYOLO, self).__init__(
      features = YOLO_Feature_Stack(),
      fc1 = L.Linear(in_size=None, out_size=1024),
      fc2 = L.Linear(in_size=1024, out_size=vec_len),
      q = L.Linear(1360, 3)
    )

  def reset_state(self):
#    self.lstm.reset_state()
    pass

  def __call__(self, x, train=True):
    out = self.features(x)
    out = self.fc1(out)
    out = F.dropout(out, self.drop_prob)
    out = F.relu(out)
    out = self.fc2(out)
#    out = F.reshape(out, self.proposal_grid_shape)
#    out = self.lstm(out)
    out = self.q(out)
    return out


class YOLO(chainer.Chain):
  def __init__(self, **params):
    # params: pgrid_dims, bb_num, num_classes

    # store params as fields
    self.__dict__.update(params)
    
    outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    self.num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    vec_len = self.num_grid_cells * outputs_per_grid_cell
    self.proposal_grid_shape = self.pgrid_dims + [outputs_per_grid_cell]

    self.to_add = np.array([1, 1, 0, 0, 1])
    self.to_mul = np.array([320, 240, 1, 1, 60])
    self.to_mul2 = np.array([1, 1, 320, 240, 1])

    super(YOLO, self).__init__(
      features = YOLO_Feature_Stack(),
      fc1 = L.Linear(in_size=None, out_size=1024),
      fc_p = L.Linear(in_size=1024, out_size=vec_len),
      fcq1 = L.Linear(in_size=vec_len, out_size=1024),
      fcq2 = L.Linear(in_size=1024, out_size=1024),
      fcq3 = L.Linear(in_size=1024, out_size=1024),
      q = L.Linear(in_size=1024, out_size=3)
    )
    self.fcq1.W.unchain_backward()


  def proposals_unr(self, x, train=True):
    out = self.features(x)
    out = self.fc1(out)
    out = F.dropout(out, self.drop_prob)
    out = F.leaky_relu(out)
    out = self.fc_p(out)
    return out     

  def proposals(self, x, train=True):
    proposals = self.proposals_unr(x, train)
    proposals = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals

  def proposals_and_q(self, x, train=True):
    proposals = self.proposals_unr(x, train)
    out = proposals
    for fc in [self.fcq1, self.fcq2, self.fcq3]:
      out = F.leaky_relu(fc(out))
    out = self.q(out)
    proposals_resh = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals_resh, out

  def __call__(self, x, train=True):
    out = self.proposals_unr(x, train)
    for fc in [self.fcq1, self.fcq2, self.fcq3]:
      out = F.leaky_relu(fc(out))
    out = self.q(out)
    return out
  
  def scale_coords(self, c):
    sc = (c + self.to_add) * self.to_mul
    sc[2:4] = [max(sc[2], 1e-3), max(sc[3], 1e-3)]
    sc[2:4] = sc[2:4]**2
    sc = sc * self.to_mul2
    return sc

  def reset_state(self):
#    self.lstm.reset_state()
    pass

class TinyYOLOQ(chainer.Chain):
  def __init__(self, **params):
    self.__dict__.update(params)
    
    outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    vec_len = num_grid_cells * outputs_per_grid_cell
    self.proposal_grid_shape = self.pgrid_dims + [outputs_per_grid_cell]

    super(TinyYOLOQ, self).__init__(
      block_1 = Conv_Pool_Block(
        [(3, 3, 16, 1)],
        [(2, 2, 2)]
      ),
      bn_1 = L.BatchNormalization(16),
      block_2 = Conv_Pool_Block(
        [(3, 3, 32, 1)],
        [(2, 2, 2)]
      ),
      bn_2 = L.BatchNormalization(32),
      block_3 = Conv_Pool_Block(
        [(3, 3, 64, 1)],
        [(2, 2, 2)]
      ),
      bn_3 = L.BatchNormalization(64),
      block_4 = Conv_Pool_Block(
        [(3, 3, 128, 1)],
        [(2, 2, 2)]
      ),
      bn_4 = L.BatchNormalization(128),
      block_5 = Conv_Pool_Block(
        [(3, 3, 256, 1)],
        [(2, 2, 2)]
      ),
      bn_5 = L.BatchNormalization(256),
      block_6 = Conv_Pool_Block(
        [(3, 3, 512, 1)],
        [(2, 2, 1)]
      ),
      bn_6 = L.BatchNormalization(512),
      block_7 = Conv_Block(
        [(3, 3, 1024, 1)]
      ),
      bn_7 = L.BatchNormalization(1024),
      block_8 = Conv_Block(
        [(3, 3, 1024, 1)]
      ),
      bn_8 = L.BatchNormalization(1024),
      block_9 = Conv_Block(
        [(1, 1, 512, 1)]
      ),
      fc1 = L.Linear(in_size=None, out_size=1024),
      fc_p = L.Linear(in_size=1024, out_size=vec_len),
      fcq1 = L.Linear(in_size=vec_len + 1024, out_size=256),
      q    = L.Linear(in_size=256, out_size=3)
   )
    self.fcq1.W.unchain_backward()

  def reset_state(self):
    pass

  def proposals_unr(self, x, train=True):
    out = self.features(x)
    out = self.fc1(out)
    out = F.dropout(out, self.drop_prob)
    pre_prop = F.leaky_relu(out)
    out = self.fc_p(pre_prop)
    return out, pre_prop

  def proposals(self, x, train=True):
    proposals, _ = self.proposals_unr(x, train)
    proposals = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals

  def proposals_and_q(self, x, train=True):
    proposals, pre_prop = self.proposals_unr(x, train)
    out = proposals
    out = F.leaky_relu(self.fcq1(F.concat((out, pre_prop))))
    out = self.q(out)
    proposals_resh = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals_resh, out

  def __call__(self, x, train=True):
    out, pre_prop = self.proposals_unr(x, train)
    out = F.leaky_relu(self.fcq1(F.concat((out, pre_prop))))
    out = self.q(out)
    return out

  def features(self, x, train=True):
    out = self.bn_1(self.block_1(x))
    out = self.bn_2(self.block_2(out))
    out = self.bn_3(self.block_3(out))
    out = self.bn_4(self.block_4(out))
    out = self.bn_5(self.block_5(out))
    out = self.bn_6(self.block_6(out))
    out = self.bn_7(self.block_7(out))
    out = self.bn_8(self.block_8(out))
    out = self.block_9(out)
    return out

class TinyYOLO(chainer.Chain):
  def __init__(self, **params):
    self.__dict__.update(params)
    
    outputs_per_grid_cell = ((self.bb_num * 7) + self.num_classes)
    num_grid_cells = self.pgrid_dims[0] * self.pgrid_dims[1]
    vec_len = num_grid_cells * outputs_per_grid_cell
    self.proposal_grid_shape = self.pgrid_dims + [outputs_per_grid_cell]

    super(TinyYOLO, self).__init__(
      block_1 = Conv_Pool_Block(
        [(3, 3, 16, 1)],
        [(2, 2, 2)]
      ),
      bn_1 = L.BatchNormalization(16),
      block_2 = Conv_Pool_Block(
        [(3, 3, 32, 1)],
        [(2, 2, 2)]
      ),
      bn_2 = L.BatchNormalization(32),
      block_3 = Conv_Pool_Block(
        [(3, 3, 64, 1)],
        [(2, 2, 2)]
      ),
      bn_3 = L.BatchNormalization(64),
      block_4 = Conv_Pool_Block(
        [(3, 3, 128, 1)],
        [(2, 2, 2)]
      ),
      bn_4 = L.BatchNormalization(128),
      block_5 = Conv_Pool_Block(
        [(3, 3, 256, 1)],
        [(2, 2, 2)]
      ),
      bn_5 = L.BatchNormalization(256),
      block_6 = Conv_Pool_Block(
        [(3, 3, 512, 1)],
        [(2, 2, 1)]
      ),
      bn_6 = L.BatchNormalization(512),
      block_7 = Conv_Block(
        [(3, 3, 1024, 1)]
      ),
      bn_7 = L.BatchNormalization(1024),
      block_8 = Conv_Block(
        [(3, 3, 1024, 1)]
      ),
      bn_8 = L.BatchNormalization(1024),
      block_9 = Conv_Block(
        [(1, 1, 512, 1)]
      ),
      fc1 = L.Linear(in_size=None, out_size=1024),
      fc_p = L.Linear(in_size=1024, out_size=vec_len),
      fcq1 = L.Linear(in_size=vec_len, out_size=1024),
      fcq2 = L.Linear(in_size=None, out_size=512),
      fcq3 = L.Linear(in_size=None, out_size=256),
      q    = L.Linear(in_size=None, out_size=3)
    )
    self.fcq1.W.unchain_backward()

  def reset_state(self):
    pass

  def proposals_unr(self, x, train=True):
    out = self.features(x)
    out = self.fc1(out)
    out = F.dropout(out, self.drop_prob)
    out = F.leaky_relu(out)
    out = self.fc_p(out)
    return out     

  def proposals(self, x, train=True):
    proposals = self.proposals_unr(x, train)
    proposals = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals

  def proposals_and_q(self, x, train=True):
    proposals = self.proposals_unr(x, train)
    out = proposals
    for fc in [self.fcq1, self.fcq2, self.fcq3]:
      out = F.leaky_relu(fc(out))
    out = self.q(out)
    proposals_resh = F.reshape(proposals, [x.shape[0]] + self.proposal_grid_shape)
    return proposals_resh, out

  def __call__(self, x, train=True):
    out = self.proposals_unr(x, train)
    for fc in [self.fcq1, self.fcq2, self.fcq3]:
      out = F.leaky_relu(fc(out))
    out = self.q(out)
    return out

  def features(self, x, train=True):
    out = self.bn_1(self.block_1(x))
    out = self.bn_2(self.block_2(out))
    out = self.bn_3(self.block_3(out))
    out = self.bn_4(self.block_4(out))
    out = self.bn_5(self.block_5(out))
    out = self.bn_6(self.block_6(out))
    out = self.bn_7(self.block_7(out))
    out = self.bn_8(self.block_8(out))
    out = self.block_9(out)
    return out



class YOLO_Feature_Stack(chainer.Chain):
  def __init__(self):
    super(YOLO_Feature_Stack, self).__init__(
      block_1 = Conv_Pool_Block(
        [(7, 7, 64, 2)],
        [(2, 2, 2)]
      ),
      block_2 = Conv_Pool_Block(
        [(3, 3, 192)],
        [(2, 2, 2)]
      ),
      block_3 = Conv_Pool_Block(
        [(1, 1, 128),
         (3, 3, 256),
         (1, 1, 256),
         (3, 3, 512)],
        [(2, 2, 2)]
      ),
      block_4 = Conv_Pool_Block(
        repeat(4)([
          (1, 1, 256),
          (3, 3, 512)
        ]) + [
        (1, 1, 512),
        (3, 3, 1024)
        ],
        [(2, 2, 2)]
      ),
      block_5 = Conv_Block(
        repeat(2)([
          (1, 1, 512),
          (3, 3, 512) # 512
        ]) + [
          (3, 3, 512),   # 512
          (3, 3, 512, 2) # 512
          ],
        ),
      block_6 = Conv_Block( 
        [(3, 3, 512), #
         (3, 3, 512)] #
      )
    )
  
  def __call__(self, x, train=True):
    out = self.block_1(x)
    out = self.block_2(out)
    out = self.block_3(out)
    out = self.block_4(out)
    out = self.block_5(out)
    out = self.block_6(out)
    return out
    
class Conv_Block(chainer.ChainList):
  def __init__(self, conv_params):
    super(Conv_Block, self).__init__()
    for i, cp in enumerate(conv_params):
      ksize = cp[0:2]
      n_filt = cp[2]
      if len(cp) == 4:
        strides = cp[3]
      else:
        strides = 1
      pad = [(k - 1) // 2 for k in ksize]
      conv = L.Convolution2D(
        in_channels=None,
        out_channels=n_filt,
        ksize=ksize,
        stride=strides,
        pad=pad
      )
      self.add_link(conv)
  
  def __call__(self, x):
    out = x
    for layer in self:
      out = F.leaky_relu(layer(x))
    return out

class Conv_Pool_Block(Conv_Block):
  def __init__(self, conv_params, max_pool_params):
    super(Conv_Pool_Block, self).__init__(conv_params)
    mp = max_pool_params[0]
    self.mp_ksize = (mp[0], mp[1])
    self.mp_strides = (mp[2], mp[2])

  def __call__(self, x):
    out = Conv_Block.__call__(self, x)
    out = F.max_pooling_2d(out, self.mp_ksize, self.mp_strides)
    return out

class Q(chainer.Chain):
    def __init__(self, width=150, height=112, channel=3, action_size=100, latent_size=100):
        feature_width = width
        feature_height = height
        for i in range(4):
            feature_width = (feature_width + 1) // 2
            feature_height = (feature_height + 1) // 2
        feature_size = feature_width * feature_height * 64
        super(Q, self).__init__(
            conv1 = L.Convolution2D(channel, 16, 8, stride=4, pad=3),
            conv2 = L.Convolution2D(16, 32, 5, stride=2, pad=2),
            conv3 = L.Convolution2D(32, 64, 5, stride=2, pad=2),
            lstm  = L.LSTM(feature_size, latent_size),
            q     = L.Linear(latent_size, action_size),
        )
        self.width = width
        self.height = height
        self.latent_size = latent_size

    def __call__(self, x, train=True):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.lstm(h3)
        q = self.q(h4)
        return q

    def reset_state(self):
        self.lstm.reset_state()
