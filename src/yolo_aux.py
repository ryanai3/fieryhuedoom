#!/usr/bin/python2

import numpy as np
import random
from random import random as rand

import chainer
from chainer import functions as F
from chainer import cuda, Variable, optimizers, serializers

import gen_data as gd

from util import *

class DQN():
  def __init__(self, params):
    # params: gpu, input_f, output_f, 
    #         pool_size, epsilon, min_epsilon, train_term, q_freeze_interval, num_epochs, gamma, batch_size
    #         num_par_games
    #         height, width, channels
    #         actions
    #         target_q
    #         optimizer, gradient_clip

    self.__dict__.update(params)
    self.move_to_gpu_if_able()
    self.gen_replay_buf()
    print "init replay buf"
    self.setup_game()
    print "game setup"
    self.action_q = self.target_q.copy()
    self.prep_optimizers()
    self.frame = 0
    self.object_id_map = {
      'MarineChainsaw': 0,
      'Demon': 1,
      'Blood': 2
    }
    random.seed()
    self.to_add = np.array([1, 1, 0, 0, 1])
    self.to_mul = np.array([320, 240, 1, 1, 60])
    self.to_mul2 = np.array([1, 1, 320, 240, 1])

  def move_to_gpu_if_able(self):
    gpu_device = None
    self.xp = np
    if self.gpu >= 0:
      cuda.check_cuda_available()
      gpu_device = self.gpu
      cuda.get_device(gpu_device).use()
      self.xp = cuda.cupy
      self.target_q.to_gpu()

  def gen_replay_buf(self):
    state_pool_shape = (self.pool_size, self.channels, self.height, self.width)
    self.state_pool = np.zeros(state_pool_shape, dtype=np.int8)
    self.action_pool = np.zeros((self.pool_size,), dtype=np.int32)
    self.reward_pool = np.zeros((self.pool_size,), dtype=np.float32)
    self.terminal_pool = np.zeros((self.pool_size,), dtype=np.float32)
    self.aux_pool = np.zeros((self.pool_size,), dtype=np.object)
    # allocate memory:
    self.state_pool[...] = 0
    self.action_pool[...] = 0
    self.reward_pool[...] = 0
    self.terminal_pool[...] = 0
#    self.aux_pool[...] = 0

  def setup_game(self):
#   self.games = [gd.setup_game() for i in range(self.num_par_games)]
    self.doom_game = gd.setup_game()

  def prep_optimizers(self):
    self.optimizer_q.setup(self.action_q)
    self.optimizer_q.add_hook(chainer.optimizer.GradientClipping(self.gradient_clip))
    self.optimizer_yolo.setup(self.action_q)

  def randomize_action(self, best, random_probability):
    if rand() < self.epsilon:
      return random.sample(self.actions, 1)[0]
    return best

  def run_episodes(self, nframes):
    rewards = []
    gs_t = self.doom_game.get_state()
    s_t = gs_t.screen_buffer.transpose((2, 0, 1))
    s_t_gpu = Variable(self.xp.asarray(np.expand_dims(s_t.astype(np.float32), 0)), volatile=True)
    action = [0, 0, 0] # just an initialization
    old_health, old_ammo = 100, 26
    for t in range(self.frame, nframes + self.frame):
      index = t % self.pool_size
      if index % 1000 == 0:
        print(index)
      # epsilon greedy get next action
      if rand() < self.epsilon: # pick random action
        action = random.sample(self.actions, 1)[0]
      else: # pick best action
        s_t_gpu = (Variable(
          self.xp.asarray(np.expand_dims(s_t.astype(np.float32), 0)),
          volatile = True
        )/(127.5)) - 1
        p_qs = self.action_q(s_t_gpu, train=False)
        action = self.actions[int(F.argmax(p_qs).data)]
      # observe reward from action 
      r_t = self.doom_game.make_action(action)

      # scale reward, mixing in health & ammo 
      new_ammo, new_health = gs_t.game_variables[0:2]
      delta_health = new_health - old_health
      delta_ammo = new_ammo - old_ammo
      old_health, old_ammo = new_health, new_ammo      
#      reward += (0.05 * delta_health) + (0.02 * delta_ammo)
      reward = r_t #+ delta_ammo * 0.02

      labels_buf = gs_t.labels_buffer
      unique_vals = np.unique(labels_buf)
      rects = []
      obj_ids = []
      zs = []
      for uv in unique_vals:
        for l in gs_t.labels:
          if l.value == uv:
            rects.append(self.get_bb(labels_buf, uv))
            zs.append(np.mean(gs_t.depth_buffer[labels_buf == uv]))
            obj_ids.append(self.object_id_map[l.object_name])
            break

      numx = self.action_q.pgrid_dims[0]
      numy = self.action_q.pgrid_dims[1]
      grid = np.zeros((numx, numy), dtype=np.dtype('O'))
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

        cell_x_min = self.find_cell(xcenter, numx)
        cell_y_min = self.find_cell(ycenter, numy)
        # care about center relative to box min
        xcenter -= cell_x_min
        ycenter -=cell_y_min

        xidx = self.find_idx(cell_x_min, numx)
        yidx = self.find_idx(cell_y_min, numy)

        coords = np.array([xcenter, ycenter, math.sqrt(xmax - xmin), math.sqrt(ymax- ymin), z/ 60 - 1], dtype=np.float32)
        cur = grid[xidx, yidx]
        if cur == 0:
          classes = np.zeros((3,), dtype=np.float32)
          classes[obj_id] = 1
          grid[xidx, yidx] = (coords.reshape((1,5)), classes)
        else:
          cur[1][obj_id] += 1 
          grid[xidx, yidx] = (np.vstack((cur[0], coords)), cur[1])

      for j in range(numx):
        for k in range(numy):
          if grid[j, k] == 0:
            grid[j, k] = (None, np.zeros((3,), dtype=np.float32))

      

      # save state_t, action, reward_t, to pool 
      self.state_pool[index] = s_t
      self.action_pool[index] = self.actions.index(action)
      self.reward_pool[index] = reward
      self.terminal_pool[index] = 0
      self.aux_pool[index] = grid

      # if this moves us into terminal state, 
      # handle moving to fresh state for next call
      if self.doom_game.is_episode_finished():
        self.doom_game.new_episode()
        old_health, old_ammo = 100, 26
        self.terminal_pool[index] = 1
        self.action_q.reset_state()
      # prep for next state
      gs_t = self.doom_game.get_state()
      s_t = gs_t.screen_buffer.transpose((2, 0, 1))
      # metadata
      rewards.append(reward)
    #  if t % 100 == 0:
   #     print(t - self.frame)
    avg_reward = sum(rewards)/len(rewards)
    max_reward = max(rewards)
    print "avg_reward", avg_reward
    print "max_reward", max(rewards)
    print "min_reward", min(rewards)
    self.frame += nframes

     
  def get_bb(self, buf, value):
    y, x = np.where(buf == value)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    return np.array(
      [[x_min, y_min],
       [x_max, y_max]]
    )

  def find_cell(self, x, y):
    return math.floor(x * y/2) / (y/2)

  def find_idx(self, x, y):
    return x * (y/2) + (y/2)

  def scale_coords(self, c, x_i, y_i):
    sc = (c + self.to_add) * self.to_mul
    sc[2:4] = [max(sc[2], 1e-3), max(sc[3], 1e-3)]
    sc[2:4] = sc[2:4]**2
    sc = sc * self.to_mul2
    sc[0] = (c[0] - 320) + (64 * x_i)
    sc[1] = (c[1] - 240) + (60 * y_i)
    return sc

  def bb_loss(self, proposals, aux_j):
    loss = Variable(self.xp.array(0.0).astype(self.xp.float32))
#    proposals = self.action_q.proposals(s_j)
    proposals_data = cuda.to_cpu(proposals.data)
    pgrid_dims = self.action_q.pgrid_dims
    bb_num = self.action_q.bb_num
    num_classes = self.action_q.num_classes
    batch_sum_iou = 0.0
    n_obj = 0
    errs = np.array([[0.0] for i in range(5)]).transpose()
    cerrs = 0
    for example in range(proposals.shape[0]):
      for x_i in range(self.action_q.pgrid_dims[0]):
        for y_i in range(self.action_q.pgrid_dims[1]):
          cell_prop = proposals[example, x_i, y_i, :]
          cell_data = proposals_data[example, x_i, y_i, :]
          aux = aux_j[example][x_i, y_i]
          if aux[0] == None: # No bounding boxes to predict
            for b in range(bb_num):
              loss += self.no_obj * (cell_prop[b * 7] ** 2)
            continue
          y_classes = aux[1]
          p_classes = cell_prop[-num_classes:]
          p_classes_cpu = cell_data[-num_classes:]
          y_classes_gpu = Variable(self.xp.asarray(y_classes))
          loss += F.sum((y_classes_gpu - p_classes)**2)
          cerrs += (p_classes_cpu - y_classes)**2
          y_obj_coords = aux[0]
          y_obj_coords_gpu = Variable(self.xp.asarray(y_obj_coords))
          n_obj += y_obj_coords.shape[0]
          for obj_i in range(y_obj_coords.shape[0]):
            y_coords = self.scale_coords(y_obj_coords[obj_i], x_i, y_i)
            best_iou = 0
            best_index = -1
            best_rmse = 10000
            for b in range(bb_num):
              p_coords = self.scale_coords(cell_data[b * 7 + 1:b * 7 + 6], x_i, y_i)
              iou = box_iou(p_coords, y_coords)
              if best_iou > 0 or iou > 0:
                if iou > best_iou:
                  best_index = b
                  best_iou = iou
              else:
                rmse = box_rmse(p_coords, y_coords)
                if rmse < best_rmse:
                  best_index = b
                  best_rmse = rmse
            batch_sum_iou += best_iou
            print(best_iou)
            best_p_coords = cell_prop[best_index * 7 + 1: best_index * 7 + 6]
#           best_p_coords = cell_prop[1:6]
            p_coords = self.scale_coords(cell_data[best_index * 7 + 1:best_index * 7 + 6], x_i, y_i)
            errs += np.abs(y_coords - p_coords)
#            print y_coords
#            print p_coords
            y_coords_gpu = y_obj_coords_gpu[obj_i]
            loss += self.coord * F.sum((best_p_coords - y_coords_gpu) ** 2)
#            loss += (1 - cell_prop[0]) ** 2
            loss += (1 - cell_prop[best_index * 7]) ** 2
#            print errs/max(n_obj, 1e-6)
#            print n_obj
    return (loss, batch_sum_iou/max(1, n_obj), errs/max(n_obj, 1e-6))
 
  def train_batches(self, num_batches):
    j =  \
      np.random.permutation(min(self.frame, self.pool_size - (self.train_term + 1)))[:self.batch_size] % self.pool_size
    gen_q = []
    self.target_q.reset_state()
    for i in range(num_batches):
      gen_q.append(self._train_batch(j + i))
    self.target_q.reset_state()
    print "Q", np.mean(gen_q)

  def _train_batch(self, j):
    j1 = j + 1
    s_j = (Variable(self.xp.asarray(self.state_pool[j].astype(np.float32)))/127.5) - 1
    s_j1 = (Variable(self.xp.asarray(self.state_pool[j + 1].astype(np.float32)))/127.5) - 1
    Qhat = self.target_q(s_j1, train=False)
    max_Q = cuda.to_cpu(F.max(Qhat, axis=1).data)
#    max_Q = cuda.to_cpu(self.xp.max(Qhat.data, axis=1))
    y_j = Variable(self.xp.asarray(self.reward_pool[j] + (1 - self.terminal_pool[j]) * self.gamma * max_Q))
    a_j = Variable(self.xp.asarray(self.action_pool[j]))
    qs = self.action_q(s_j)
    q_preds = F.select_item(qs, a_j)
    q_loss = F.mean_squared_error(y_j, q_preds)
    self.optimizer.zero_grads()
    res = q_loss.backward()
    q_loss.unchain_backward()
    self.optimizer.update()
    qp_cpu = qs.data
#    print "loss", loss.data
#    print np.mean(qp_cpu, axis=0)
#    print(res)
    return np.mean(cuda.to_cpu(q_preds.data))
#    print "Q", np.mean(q_preds.data)
#    print "loss", loss.data
#    print np.mean(qp_cpu, axis=0)

  def train_batch(self):
    j =  \
      np.random.permutation(min(self.frame, self.pool_size - self.train_term))[:self.batch_size] % self.pool_size
    j1 = j + 1
    s_j = (Variable(self.xp.asarray(self.state_pool[j].astype(np.float32)))/127.5) - 1
    s_j1 = (Variable(self.xp.asarray(self.state_pool[j + 1].astype(np.float32)))/127.5) - 1
    Qhat = self.target_q(s_j, train=False)
#   Qhat = self.target_q(s_j1, train=False)
    max_Q = cuda.to_cpu(F.max(Qhat, axis=1).data)
#    import pdb; pdb.set_trace()
#    max_Q = cuda.to_cpu(self.xp.max(Qhat.data, axis=1))
    y_j = Variable(self.xp.asarray(self.reward_pool[j] + (1 - self.terminal_pool[j]) * self.gamma * max_Q))
    a_j = Variable(self.xp.asarray(self.action_pool[j]))
    
    proposals, qs = self.action_q.proposals_and_q(s_j, train=True)
    q_preds = F.select_item(qs, a_j)
    bb_loss, b_s_iou, errs = self.bb_loss(proposals, self.aux_pool[j])
    q_loss = F.mean_squared_error(y_j, q_preds)
#    import pdb; pdb.set_trace()
    self.optimizer_yolo.zero_grads()
    self.optimizer_q.zero_grads()
    bb_loss.backward()
    q_loss.backward()
    self.optimizer_yolo.update()
    self.optimizer_q.update()
    
    qp_cpu = qs.data
#    print "Q", np.mean(q_preds.data)
#    print "loss", loss.data
#    print "b_s_iou", b_s_iou
#    print np.mean(qp_cpu, axis=0)
    return bb_loss.data, np.mean(q_preds.data), b_s_iou, errs

  def run(self):
    self.run_episodes(self.pool_size)
#    self.add_to_replay_buf(self.pool_size)
    for epoch in range(self.num_epochs):
      print("EPOCH!")
      for b_i in range(self.batches_per_epoch / self.train_term):
        if (b_i * self.train_term) % self.q_freeze_interval == 0:
          print "============"
          print b_i, "unfreeze!"
          print "============"
#          self.action_q = self.target_q.copy()
          self.target_q = self.action_q.copy()
          self.action_q.reset_state()
        if b_i % self.batches_per_replay_update == 0 and b_i != 0:
          print(b_i), "updating replay!"
          self.action_q.reset_state()
          self.run_episodes(self.frames_per_replay_update)
          self.action_q.reset_state()
#          self.action_q.reset_state()
  #        self.add_to_replay_buf(self.frames_per_replay_update)
#        self.train_batches(self.train_term)
        bb_loss, avg_q, b_s_iou, errs = self.train_batch()
        if b_i % 10 == 0:
          print(b_i)
          for s, v in zip("x y w h z".split(), errs.transpose()):
            print s + "_diff", v
          print "b_s_iou", b_s_iou
          print "bb_loss", bb_loss
          print "avg_q", avg_q
      self.epsilon = self.epsilon * 0.98
      self.action_q.reset_state()
      chainer.serializers.save_hdf5("/data/r9k/past_runs/tiny_yoloq3_{0}".format(epoch), self.action_q)
  
import argparse
from chainer_dqn_net import Q

from nets import *

if __name__ == "__main__":
  # params: gpu, input_f, output_f, 
    #         pool_size, epsilon, min_epsilon, train_term, q_freeze_interval, num_epochs, gamma, batch_size
    #         num_par_games
    #         height, width, channels
    #         actions
    #         target_q
    #         optimizer, gradient_clip

  params =  {
    'gpu': 0,
    'pool_size': 4 * 1000, # x * 1000
    'epsilon': 0.2,
    'train_term': 1,
    'q_freeze_interval': 500,
    'num_epochs': 100,
    'gamma': 0.95,
    'batch_size': 8,
    'batches_per_epoch': 200, #1000, #100,
    # 150/min -> 1500 = 10 min, 15,000 = 1hr 40min, 
    'batches_per_replay_update': 2500, #10,
    'frames_per_replay_update': 4 * 1000,
    'height': 480,
    'width': 640,
    'channels': 3,
    'actions': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'target_q': Q(width=640, height=480, channel=3, action_size=3),
    'optimizer_q': optimizers.AdaGrad(lr = (10 ** -6)/8),
    'gradient_clip': 0.01,
    'optimizer_yolo': optimizers.AdaGrad(lr = 10 ** -2),
    # yolo params (hacky):
    'no_obj': 4.0,
    'coord': 5.0,
  }

  params['target_q'] = TinyYOLOQ(**{'pgrid_dims': [10, 8], 'bb_num': 3, 'num_classes': 3, 'drop_prob': 0.1})
  dqn = DQN(params)
#  import pdb; pdb.set_trace()
  dqn.run()
#  import pdb; pdb.set_trace()
      
