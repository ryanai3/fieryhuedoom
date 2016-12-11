#!/usr/bin/python2

import numpy as np
import random
from random import random as rand

import chainer
from chainer import functions as F
from chainer import cuda, Variable, optimizers, serializers

import gen_data as gd

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
    self.prep_optimizer()
    self.frame = 0
    random.seed()


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
    self.aux_pool = np.zeros((self.pool_size,), dtype=np.float32)
    # allocate memory:
    self.state_pool[...] = 0
    self.action_pool[...] = 0
    self.reward_pool[...] = 0
    self.terminal_pool[...] = 0
    self.aux_pool[...] = 0

  def setup_game(self):
#   self.games = [gd.setup_game() for i in range(self.num_par_games)]
    self.doom_game = gd.setup_game()

  def prep_optimizer(self):
    self.optimizer.setup(self.action_q)
    self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.gradient_clip))

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

      # save state_t, action, reward_t, to pool 
      self.state_pool[index] = s_t
      self.action_pool[index] = self.actions.index(action)
      self.reward_pool[index] = reward
      self.terminal_pool[index] = 0
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
    loss = F.mean_squared_error(y_j, q_preds)
    self.optimizer.zero_grads()
    res = loss.backward()
    loss.unchain_backward()
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
    Qhat = self.target_q(s_j1, train=False)
    max_Q = cuda.to_cpu(F.max(Qhat, axis=1).data)
#    max_Q = cuda.to_cpu(self.xp.max(Qhat.data, axis=1))
    y_j = Variable(self.xp.asarray(self.reward_pool[j] + (1 - self.terminal_pool[j]) * self.gamma * max_Q))
    a_j = Variable(self.xp.asarray(self.action_pool[j]))
    qs = self.action_q(s_j)
    q_preds = F.select_item(qs, a_j)
    loss = F.mean_squared_error(y_j, q_preds)
    self.optimizer.zero_grads()
    loss.backward()
    loss.unchain_backward()
    self.optimizer.update()
    qp_cpu = qs.data
    print "Q", np.mean(q_preds.data)
    print "loss", loss.data
    print np.mean(qp_cpu, axis=0)

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
        if b_i % self.batches_per_replay_update == 0:
          print(b_i), "updating replay!"
          self.action_q.reset_state()
          self.run_episodes(self.frames_per_replay_update)
          self.action_q.reset_state()
#          self.action_q.reset_state()
  #        self.add_to_replay_buf(self.frames_per_replay_update)
#        self.train_batches(self.train_term)
        self.train_batch()
      self.epsilon = self.epsilon * 0.98
      self.action_q.reset_state()
      chainer.serializers.save_hdf5("/data/r9k/past_runs/fancy_dqn_{0}".format(epoch), self.action_q)
  
import argparse
from chainer_dqn_net import Q

from nets import ControlYOLO

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
    'pool_size': 25 * 1024,
    'epsilon': 0.2,
    'train_term': 1,
    'q_freeze_interval': 500,
    'num_epochs': 100,
    'gamma': 0.95,
    'batch_size': 16,
    'batches_per_epoch': 2500, #1000, #100,
    # 150/min -> 1500 = 10 min, 15,000 = 1hr 40min, 
    'batches_per_replay_update': 10,
    'frames_per_replay_update': 100,
    'height': 480,
    'width': 640,
    'channels': 3,
    'actions': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'target_q': Q(width=640, height=480, channel=3, action_size=3),
    'optimizer': optimizers.AdaGrad(lr = 10 ** -6),
    'gradient_clip': 0.01
  }

  params['target_q'] = ControlYOLO(**{'pgrid_dims': [10, 8], 'bb_num': 1, 'num_classes': 10, 'drop_prob': 0.5})
  dqn = DQN(params)
#  import pdb; pdb.set_trace()
  dqn.run()
#  import pdb; pdb.set_trace()
      
