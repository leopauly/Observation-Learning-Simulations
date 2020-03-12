#### Reward Evaluation
## Wrtitten by : leopauly | cnlp@leeds.ac.uk
## Courtesy for DDPG implementation : Steven Spielberg Pon Kumar (github.com/stevenpjg)
####

##Imports
import gym
from gym.spaces import Box, Discrete
import numpy as np
np.set_printoptions(suppress=True)
import cv2
from ddpg import DDPG
import matplotlib.pyplot as plt
import scipy.misc as misc
import os
from threading import Thread, Lock
import sys
from six.moves import xrange
import warnings
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
import lscript as lsp
import modelling as md
from Policy import Saved_Policy
warnings.filterwarnings("ignore",category=DeprecationWarning)


## Defining env
env = gym.make('LunarLanderContinuous-v2')
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"

## Defining vars 
num_episodes=100
steps=80 # No of steps taken in a episode
is_batch_norm = False 
xrange=range 
start_training=64 
height=112 
width=112 
channel=3
crop_size=112
cluster_length=16 
nb_classes=2 
feature_size=4608
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def custom_get_eval(observed_value):
    print('observed_value',observed_value)
    lander_pos_x=np.array(observed_value[0])
    lander_pos_y=np.array(observed_value[1])
    return np.linalg.norm((lander_pos_x,lander_pos_y)-np.array((0,0)))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


def s2l(i_run):
    print('This is the ith run',i_run)

    num_states =  env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]   
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)

    agent = Saved_Policy(num_states,num_actions)
    eval_metric_st= np.array([0])
    obs_store=[]
    
    for episode in range(num_episodes):
        print ("==== Starting episode no:",episode,"====","\n")
        observation=env.reset()   
        env.render()
        obs_img=env.render(mode='rgb_array')   
        #obs_img=np.array(misc.imresize(obs_img,[112,112,3]))
        #observation =np.array(frame_obj.frame_feature_extractor(obs_img))
        #observation=observation.reshape(-1)

        

        for i in range(steps):

            x = observation

            ## Printing eval_metric after every step
            eval_metric=np.array(custom_get_eval(observation))
            eval_metric=eval_metric.reshape(-1)
            print('Distance to goal:',eval_metric)    
            eval_metric_st = np.append(eval_metric_st,eval_metric)           
            np.savetxt('test_eval_metric_per_step.txt',eval_metric_st, newline="\n")

            action = agent.get_action(np.reshape(x,[1,num_states]))
            action_x=action[0][0]
            action_y=action[0][1]
            action=[action_x,action_y]

            print ('Action at episode-',episode, 'step-', i ," :",action)


            action_max = np.array(env.action_space.high).tolist()
            action_min = np.array(env.action_space.low).tolist()
            action=np.clip(action,action_min,action_max)
            print ('Action at episode clipped-',episode, 'step-', i ," :",action)
               
            observation,_,_,_=env.step(action)
            obs_robo_=env.render(mode='rgb_array')   # Get the observation
            #obs_robo=misc.imresize(obs_robo_,[112,112,3])
            #observation=np.array(frame_obj.frame_feature_extractor(np.array(obs_robo)))
            #observation=observation.reshape(-1)
            
    
    del agent


if __name__=='__main__':
    from datetime import datetime
    start_time=str(datetime.now())
    run_start=0
    run_end=1
    print('Start trial:',run_start,'End trail:',run_end-1)
    for i_run in range(run_start,run_end):
        s2l(i_run)
    print('Start to end time:',start_time,str(datetime.now()))
