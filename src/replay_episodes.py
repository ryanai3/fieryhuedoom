#!/usr/bin/python2
import sys
from chainer import cuda, Variable, optimizers, serializers
import gen_data as gd
import argparse
from chainer_dqn_net import Q
from chainer import functions as F
from nets import *
import numpy as np
import imageio
import cv2

left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

def play_using_saved_q(q_filepath, num_episodes=1, save_replay=False, replay_filepath=None, device=0):
    d = int(device)
    cuda.get_device(d).use()
    if save_replay and not replay_filepath:
        print ("Error: please provide a filepath for replays")
    #saved_q = Q(width=640, height=480, latent_size=256, action_size=3)
    saved_q = ControlYOLO(**{'pgrid_dims': [10, 8], 'bb_num': 1, 'num_classes': 10, 'drop_prob': 0.5})
    saved_q.to_gpu(device=d)
    #import pdb; pdb.set_trace()
    serializers.load_hdf5(q_filepath, saved_q)
    doom_game = gd.setup_game(show_window=False)
    for i in range(int(num_episodes)):
        doom_game.new_episode(replay_filepath + str(i) + "_rec.lmp")
        total_reward = 0
        while not doom_game.is_episode_finished():
            state = doom_game.get_state()
            screen_buf = cuda.to_gpu((state.screen_buffer.astype(np.float32).transpose((2, 0, 1))), device=d)
            screen_buf = Variable(screen_buf.reshape((1,) + screen_buf.shape) / 127.5 - 1, volatile=True)
            scores = saved_q(screen_buf, train=False)
            best_idx = int(F.argmax(scores).data)
            total_reward += doom_game.make_action(actions[best_idx])
        print("Total reward:", total_reward)
    doom_game.close()



def replay(replay_filepath):
    print("*****************")
    print("Replaying episode: {0}".format(replay_filepath))
    doom_game = gd.setup_game(show_window=False, show_hud=True, show_particles=True, show_weapon=True)
    doom_game.replay_episode(replay_filepath + "_rec.lmp")
    total_reward = 0
    imgs = []
    while not doom_game.is_episode_finished():
        s = doom_game.get_state()
        screen = s.screen_buffer
        imgs.append(screen)
        doom_game.advance_action()
        r = doom_game.get_last_reward()
        if r != 0:
	    print("reward: ", r)
        total_reward += r
    #import pdb; pdb.set_trace()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('/data/r9k/obs_data/{0}.avi'.format(replay_filepath), fourcc, 15, (640, 480))
    for img in imgs:
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    #imageio.mimsave("/data/r9k/obs_data/test_gif_new.gif", imgs)
    print("Episode finished. Total reward was {0}.".format(str(total_reward)))
    print("*****************")
 	

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "q":
        play_using_saved_q(*sys.argv[2:])
    elif mode == "r":
        # TODO: support multiple replays
        replay(sys.argv[2])
    else:
        print "Unknown mode. Usage:"
        print "q [q_filepath] [num_episodes - default 1] [save_replay - default False] [replay_filepath - required if save replay is true]"
        print "Plays a game using a saved Q model"
        print "r [replay_filepath]"
        print "Replays the given episode"

