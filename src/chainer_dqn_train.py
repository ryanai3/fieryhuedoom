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
import pyautogui as ag
#from game import PoohHomerun
from net import Q
import chainer
from chainer import functions as F
from chainer import cuda, Variable, optimizers, serializers
import sys

# action definition
left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

latent_size = 256
gamma = 0.99
batch_size = 64
ag.PAUSE = 0

parser = argparse.ArgumentParser(description='Deep Q-learning Network for game using mouse')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model file path without extension')
parser.add_argument('--interval', default=100, type=int,
                    help='interval of capturing (ms)')
parser.add_argument('--random', '-r', default=0.2, type=float,
                    help='randomness of play')
parser.add_argument('--pool_size', default=50000, type=int,
                    help='number of frames of memory pool size')
parser.add_argument('--random_reduction', default=0.000002, type=float,
                    help='reduction rate of randomness')
parser.add_argument('--min_random', default=0.1, type=float,
                    help='minimum randomness of play')
parser.add_argument('--train_term', default=4, type=int,
                    help='training term size')
parser.add_argument('--train_term_increase', default=0.00002, type=float,
                    help='increase rate of training term size')
parser.add_argument('--max_train_term', default=32, type=int,
                    help='maximum training term size')
parser.add_argument('--double_dqn', action='store_true',
                    help='use Double DQN algorithm')
parser.add_argument('--update_target_interval', default=2000, type=int,
                    help='interval to update target Q function of Double DQN')
parser.add_argument('--only_result', action='store_true',
                    help='use only reward to evaluate')
args = parser.parse_args()

interval = args.interval / 1000.0
only_result = args.only_result
update_target_interval = args.update_target_interval

#game = PoohHomerun()
#game.load_images('image')
if game.detect_position() is None:
    print "Error: cannot detect game screen position."
    exit()
train_width = 640 / 4
train_height = 480 / 4
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
state_pool = np.zeros((POOL_SIZE, 3, train_height, train_width), dtype=np.float32)
action_pool = np.zeros((POOL_SIZE,), dtype=np.int32)
reward_pool = np.zeros((POOL_SIZE,), dtype=np.float32)
terminal_pool = np.zeros((POOL_SIZE,), dtype=np.float32)

# allocate memory
state_pool[...] = 0
action_pool[...] = 0
reward_pool[...] = 0
terminal_pool[...] = 0

if only_result:
    terminal_pool[-1] = 1
frame = 0
average_reward = 0

optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-06)
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
    if use_double_dqn:
        target_q = q.copy()
        target_q.reset_state()
    while True:
        term_size = int(current_term_size)
        if frame < batch_size * term_size:
            continue
        batch_index = np.random.permutation(min(frame - term_size, POOL_SIZE))[:batch_size]
        train_image = Variable(xp.asarray(state_pool[batch_index]))
        y = q(train_image)
        
        for term in range(term_size):
            next_batch_index = (batch_index + 1) % POOL_SIZE
            train_image = Variable(xp.asarray(state_pool[next_batch_index]))
            score = q(train_image)
            if only_result:
                t = Variable(xp.asarray(reward_pool[batch_index]))
            else:
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
            print "loss", float(cuda.to_cpu(loss.data))
            clock = time.clock()
            print "train", clock - last_clock
            last_clock = clock
        current_term_size = min(current_term_size * term_increase_rate, max_term_size)
        print "current_term_size ", current_term_size

if __name__ == '__main__':
    try:
	doom_game = gd.setup_game()
        thread.start_new_thread(train, ())
        next_clock = time.clock() + interval
        save_iter = 10000
        save_count = 0
        action = None
        action_q = q.copy()
        action_q.reset_state()
        while True:
            if action is not None:
                #game.play(action)
		reward = doom_game.make_action(action)
# convert this crap to pull img from vizdoom
            state = doom_game.get_state()
            train_image = state.screen_buffer().astype(np.float32).transpose((2, 0, 1))

            #reward, terminal = game.process(screen)
            terminal = doom_game.is_episode_finished()
            if reward is not None:
                
                train_image = Variable(train_image, volatile=True)
                score = action_q(train_image, train=False)
                best_idx = int(np.argmax(score.data))

                # action = game.randomize_action(best, random_probability)
		action = randomize_action(actions[best_idx], random_probability)
                print action, float(score.data[0][action]), best, float(score.data[0][best]), reward
                index = frame % POOL_SIZE
                state_pool[index] = cuda.to_cpu(train_image.data)
                action_pool[index] = action
                reward_pool[index - 1] = reward
                average_reward = average_reward * 0.9999 + reward * 0.0001
                print "average reward: ", average_reward
                if terminal:
                    terminal_pool[index - 1] = 1
                    if only_result:
                        i = index - 2
                        r = reward
                        while terminal_pool[i] == 0:
                            r = reward_pool[i] + gamma * r
                            reward_pool[i] = r
                            i -= 1
                    action_q = q.copy()
                    action_q.reset_state()
                else:
                    terminal_pool[index - 1] = 0
                frame += 1
                save_iter -= 1
                random_probability *= random_reduction_rate
                if random_probability < min_random_probability:
                    random_probability = min_random_probability
            else:
                action = None
                if save_iter <= 0:
                    print 'save: ', save_count
                    serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, save_count), q)
                    serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, save_count), optimizer)
                    save_iter = 10000
                    save_count += 1
            current_clock = time.clock()
            wait = next_clock - current_clock
            print 'wait: ', wait
            if wait > 0:
                next_clock += interval
                time.sleep(wait)
            elif wait > -interval / 2:
                next_clock += interval
            else:
                next_clock = current_clock + interval
    except KeyboardInterrupt:
        pass
