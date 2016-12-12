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
import math

left  = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]

actions = [shoot, left, right]

def play_using_saved_q(q_filepath, num_episodes=1, save_replay=False, replay_filepath=None, device=0):
    #d = int(device)
    #cuda.get_device(d).use()
    if save_replay and not replay_filepath:
        print ("Error: please provide a filepath for replays")
    #saved_q = Q(width=640, height=480, latent_size=256, action_size=3)
    #saved_q = ControlYOLO(**{'pgrid_dims': [10, 8], 'bb_num': 1, 'num_classes': 10, 'drop_prob': 0.5})
    saved_q = YOLO(**{'pgrid_dims': [10, 8], 'bb_num': 3, 'num_classes': 3, 'drop_prob': 0.5})
    #saved_q.to_gpu(device=d)
    #import pdb; pdb.set_trace()
    serializers.load_hdf5(q_filepath, saved_q)
    doom_game = gd.setup_game(show_window=False)
    for i in range(int(num_episodes)):
        doom_game.new_episode(replay_filepath + str(i) + "_rec.lmp")
        total_reward = 0
        ct = 0
        while not doom_game.is_episode_finished():
            ct+=1
            if ct % 10 == 0:
              print ct
            state = doom_game.get_state()
            #screen_buf = cuda.to_gpu((state.screen_buffer.astype(np.float32).transpose((2, 0, 1))), device=d)
            screen_buf = state.screen_buffer.astype(np.float32).transpose((2, 0, 1))
            screen_buf = Variable(screen_buf.reshape((1,) + screen_buf.shape) / 127.5 - 1, volatile=True)
            scores = saved_q(screen_buf, train=False)
            best_idx = int(F.argmax(scores).data)
            total_reward += doom_game.make_action(actions[best_idx])
        print("Total reward:", total_reward)
    doom_game.close()

def play_draw_and_record_yolo(yolo_filepath, replay_filepath, num_episodes=1, device=0):
    d = int(device)
    cuda.get_device(d).use()

    yolo = YOLO(**{'pgrid_dims': [10, 8], 'bb_num': 3, 'num_classes': 3, 'drop_prob': 0.5})
    yolo.to_gpu()
    serializers.load_hdf5(yolo_filepath, yolo)
    
    doom_game = gd.setup_game(show_window=False)
    for i in range(int(num_episodes)):
        doom_game.new_episode(replay_filepath + str(i) + "_rec.lmp")
        total_reward = 0
        while not doom_game.is_episode_finished():
            state = doom_game.get_state()
            screen_buf = cuda.to_gpu((state.screen_buffer.astype(np.float32).transpose((2, 0, 1))), device=d)
            screen_buf = Variable(screen_buf.reshape((1,) + screen_buf.shape) / 127.5 - 1, volatile=True)
            grid_var, scores = yolo.proposals_and_q(screen_buf, train=False)
            best_idx = int(F.argmax(scores).data)
            total_reward += doom_game.make_action(actions[best_idx])

            grid = cuda.to_cpu(grid_var.data[0])
            
            
            boxes = []
            base_img = doom_game.get_state().screen_buffer
            """
            for x, y in np.ndindex((10,8)):
                proposals = grid[x, y]
                class_probs = proposals[20:]
                best_class = class_probs.index(max(class_probs))
                for c in range(3):
                    conf_idx = c * 7
                    if proposals[conf_idx]: # >= 0.6:
                        scaled = yolo.scale_coords(proposals[c+1:c+6])
                        box = (scaled, x, y, best_class)
                        boxes.append(box)
            """
            scaled = yolo.scale_coords(np.array([ 0.1921875 ,  0.03125   ,  0.18540496,  0.3354102 ,  0.66666667]))
            boxes.append((scaled, 4, 4, 1))
                
                    
            import pdb; pdb.set_trace()
            # sort boxes by confidence
            for box in boxes:
                w = box[0][2]
                h = box[0][3]
                xcenter = box[0][0] - 320 + box[1] * 64
                xmin = int(round(xcenter - w/2))
                xmax = int(round(xcenter + w/2))
                ycenter = box[0][1] - 240 + box[2] * 60
                ymin = int(round(ycenter - h/2))
                ymax = int(round(ycenter + h/2))
                z_ = round(box[0][4])
                best_class = box[3]
                #import pdb; pdb.set_trace()
                # draw the bounding boxes
                for x_ in range(xmin, xmax+1):
                    base_img[ymin, x_, best_class] = 255
                    base_img[ymax, x_, best_class] = 255
                for y_ in range(ymin, ymax+1):
                    base_img[y_, xmin, best_class] = 255
                    base_img[y_, xmax, best_class] = 255
            import pdb; pdb.set_trace()


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
    elif mode == "y":
        play_draw_and_record_yolo(*sys.argv[2:])
    elif mode == "r":
        # TODO: support multiple replays
        replay(sys.argv[2])
    else:
        print "Unknown mode. Usage:"
        print "q [q_filepath] [num_episodes - default 1] [save_replay - default False] [replay_filepath - required if save replay is true]"
        print "Plays a game using a saved Q model"
        print "r [replay_filepath]"
        print "Replays the given episode"

