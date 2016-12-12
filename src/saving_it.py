#!/usr/bin/python2
# Simple DQN directly from the pixels
# Adapted from https://github.com/dsanno/chainer-dqn/blob/master/src/train.py 
# to work on VizDoom's Doom environment

import gen_data as gd
import argparse
import time
import thread
import os
import random
import numpy as np
#from game import PoohHomerun
from chainer_dqn_net import Q
import chainer
from chainer import functions as F
from chainer import cuda, Variable, optimizers, serializers
import sys

# action definition
left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

latent_size = 512
gamma = 0.95
batch_size = 64

#q_result_filename = 'q_results.txt'
#q_file = open(q_result_filename, 'a')

parser = argparse.ArgumentParser(description='Deep Q-learning Network for game using mouse')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--random', '-r', default=0.2, type=float,
                    help='randomness of play')
parser.add_argument('--pool_size', default=50000, type=int,
                    help='number of frames of memory pool size')
parser.add_argument('--random_reduction', default=0.000002, type=float,
                    help='reduction rate of randomness')
parser.add_argument('--min_random', default=0.1, type=float,
                    help='minimum randomness of play')
parser.add_argument('--train_term', default=10, type=int,
                    help='training term size')
parser.add_argument('--train_term_increase', default=0.00002, type=float,
                    help='increase rate of training term size')
parser.add_argument('--max_train_term', default=32, type=int,
                    help='maximum training term size')
parser.add_argument('--double_dqn', action='store_true',
                    help='use Double DQN algorithm')
parser.add_argument('--update_target_interval', default=1000, type=int,
                    help='interval to update target Q function of Double DQN')
parser.add_argument('--only_result', action='store_true',
                    help='use only reward to evaluate')
parser.add_argument('--num_epochs', default=10, type=int)
args = parser.parse_args()

update_target_interval = args.update_target_interval
num_epochs = args.num_epochs

train_width = 640
train_height = 480
random.seed()

gpu_device = None
xp = np
q = Q(width=train_width, height=train_height, latent_size=latent_size, action_size=3)
target_q = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy
    q.to_gpu()

POOL_SIZE = args.pool_size
state_pool = np.zeros((POOL_SIZE, 3, train_height, train_width), dtype=np.int8)
action_pool = np.zeros((POOL_SIZE,), dtype=np.int32)
reward_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
terminal_pool = np.zeros((POOL_SIZE,), dtype=np.float32)

# allocate memory
state_pool[...] = 0
action_pool[...] = 0
reward_pool[...] = 0
terminal_pool[...] = 0


frame = 0

doom_game = gd.setup_game()

optimizer = optimizers.Adam()#(rho=0.95, eps=1e-06)
optimizer.setup(q)
optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))
if args.input is not None:
    serializers.load_hdf5('{}.model'.format(args.input), q)
    serializers.load_hdf5('{}.state'.format(args.input), optimizer)

random_probability = args.random
random_reduction_rate = 1 - args.random_reduction
min_random_probability = min(random_probability, args.min_random)


def randomize_action(best, random_probability):
    if random.random() < random_probability:
        return random.sample(actions, 1)[0]
    return best

def train():
    max_term_size = args.max_train_term
    current_term_size = args.train_term
    term_increase_rate = 1 + args.train_term_increase
    last_clock = time.clock()
    update_target_iteration = 0
 
    term_size = int(current_term_size)
    batch_index = np.random.permutation(min(frame - term_size, POOL_SIZE))[:batch_size]
    train_image = Variable(xp.asarray(state_pool[batch_index]))
    y = q(train_image)
    #import pdb; pdb.set_trace()
    
    for term in range(term_size):
        next_batch_index = (batch_index + 1) % POOL_SIZE
        train_image = Variable(xp.asarray(state_pool[next_batch_index]))
        score = q(train_image)
        best_q = cuda.to_cpu(xp.max(score.data, axis=1))
        t = Variable(xp.asarray(reward_pool[batch_index] + (1 - terminal_pool[batch_index]) * gamma * best_q))
        action_index = chainer.Variable(xp.asarray(action_pool[batch_index]))
        loss = F.mean_squared_error(F.select_item(y, action_index), t)
        y = score
        optimizer.zero_grads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        batch_index = next_batch_index
#        print "loss", float(cuda.to_cpu(loss.data))
#        clock = time.clock()
#        print "train", clock - last_clock
#        last_clock = clock
#        current_term_size = min(current_term_size * term_increase_rate, max_term_size)
#        print "current_term_size ", current_term_size
    print "Q", xp.mean(y.data)
#    q_file.write(str(xp.mean(y.data)))
        

def fill_replay_buf(num_frames):
    average_reward = 0
    global frame
#    import pdb; pdb.set_trace()
    episode_num = 0
    action = randomize_action(left, 1)
    old_health = 100
    old_ammo = 26
    for sframe in range(frame, frame + num_frames):
        reward = doom_game.make_action(action)
        terminal = doom_game.is_episode_finished()
        if terminal:
            doom_game.new_episode()
            terminal_pool[index - 1] = 1
            old_health = 100
            old_ammo = 26
            action = randomize_action(left, 1)
        state = doom_game.get_state()
        game_vars = state.game_variables
        new_health = game_vars[1]
        delta_health = new_health - old_health
        old_health = new_health
        new_ammo = game_vars[0]
        delta_ammo = new_ammo - old_ammo
        old_ammo = new_ammo
        reward += 0.05 * delta_health
        reward += 0.02 * delta_ammo
            
        train_image = cuda.to_gpu((state.screen_buffer.astype(np.float32).transpose((2, 0, 1))), device=args.gpu)
        import pdb; pdb.set_trace()
            #reward, terminal = game.process(screen)

        train_image = Variable(train_image.reshape((1,) + train_image.shape) / 127.5 - 1, volatile=True)
        score = action_q(train_image, train=False)

        best_idx = int(F.argmax(score).data)
        

        # action = game.randomize_action(best, random_probability)
        action = randomize_action(actions[best_idx], random_probability)
        index = sframe % POOL_SIZE
        state_pool[index] = cuda.to_cpu(train_image.data)
        action_pool[index] = actions.index(action)
        reward_pool[index - 1] = reward
        average_reward = average_reward * 0.9 + reward * 0.1
	#if sframe % 100 == 0:
           # print(average_reward)
        terminal_pool[index - 1] = 0
    frame += num_frames  

if __name__ == '__main__':
    print 'game initialized'
#    thread.start_new_thread(train, ())
    save_iter = 10000
    save_count = 0
    action = left
    action_q = q.copy()
    action_q.reset_state()
    old_health = 100
    old_ammo = 26
    trains_per_epoch = 1000
    ct = 0
    w_u = 0
    epochs = 0
    print("Started training!")
    for epoch in range(num_epochs):
	print("epoch {0}".format(epoch))
        for train_event in range(trains_per_epoch):
            print("train_event! {0}".format(train_event))
            fill_replay_buf(1000)
            train()    
	random_probability = max(min_random_probability, random_probability * random_reduction_rate)


