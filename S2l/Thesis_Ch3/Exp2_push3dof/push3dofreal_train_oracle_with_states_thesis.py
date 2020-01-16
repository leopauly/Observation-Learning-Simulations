#### Training agent in Pusher3Dof gym env using a single real-world env
## Wrtitten by : leopauly | cnlp@leeds.ac.uk
## Courtesy for DDPG implementation : Steven Spielberg Pon Kumar (github.com/stevenpjg)
## Reward used : Hand crafted task completion metrics
####

##Imports
import gym
from gym.spaces import Box, Discrete
import numpy as np
np.set_printoptions(suppress=True)
import cv2
from ddpg import DDPG
from ou_noise import OUNoise
import matplotlib.pyplot as plt
import scipy.misc as misc

## Imports for DNN
import os
from threading import Thread, Lock
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
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

## Custom scripts
import lscript as lsp
import modelling as md

## Defining env
env = gym.make('Pusher3DOFReal-v1')
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"

## Defining vars for reinfrocement learning algo
num_episodes=20
steps=160 # No of actions taken in a roll out
is_batch_norm = False #batch normalization switch
xrange=range # For python3
start_training=64 # Buffer size, before starting to train the RL algorithm

## vars for feature extraction
height=112 
width=112 
channel=3
crop_size=112

cluster_length=16 # Length of one activity
nb_classes=2 
frame_feature_size=4608 #8192   #16384  #487 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Frame_Feature:
    def __init__(self):
         self.g=tf.Graph()
         with self.g.as_default():
              self.sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
              self.base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(height,width,channel), pooling=None, classes=1000)
              #print(tf.contrib.graph_editor.get_tensors(self.g))   #(tf.get_default_graph()))
              self.base_model._make_predict_function()
              print('VggNet loaded with Imagenet values')
    
    def frame_feature_extractor(self,frame_):
        frame= self.im_preprocess(frame_)
        frame=frame.reshape(-1,height,width,channel)
        frame_features=self.base_model.predict(frame)
        return frame_features

    def im_preprocess(self,im):
        im = np.float32(im)
        im[:,:,2] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,0] -= 123.68
        im = im[:, :, ::-1]  # change to BGR
        return im

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


def s2l(i_run):
    print('This is the ith run',i_run)

    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    num_states =1  #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]   
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)

    agent = DDPG(env, is_batch_norm,num_states,num_actions)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0 
    total_reward=0
    best_reward=-10000
    
    print ("Number of Steps per episode:", steps)
    reward_st_per_episode = np.array([0])  #saving reward
    eval_metric_st= np.array([0])
    eval_metric_st_per_episode= np.array([0])
    reward_st_per_step = np.array([0])  #saving reward after every step


    frame_obj=Frame_Feature()


    for episode in range(num_episodes):
        print ("==== Starting episode no:",episode,"====","\n")
        env.reset()   # Reset env in the begining of each episode
        env.render()

        initial_distance=np.array(env.get_eval())
        observation=initial_distance
        print('Initial distance before the episode:',initial_distance)
        reward_per_episode = 0

        vid_robo_=[]

        for i in range(steps):

            x = observation

            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()/(episode+1)
            print('noise',noise)
            action = action[0] + noise #Select action according to current policy and exploration noise
            print ('Action at episode-',episode, 'step-', i ," :",action)

               
            _,_,_,_=env.step(action)
            env.render()

            observation=np.array(env.get_eval())
            observation=observation.reshape(-1)
          
                
            if(i==(steps-1)):
                reward=1-(np.array(env.get_eval())/initial_distance)
                reward=np.array(reward)
                print('reward: ',reward)
            else:
                reward=0
                reward=np.array(reward)
                print('reward: ',reward)

            # Printing eval_metric after every step
            eval_metric=np.array(env.get_eval())
            eval_metric=eval_metric.reshape(-1)
            print('Distance to goal:',eval_metric)    
            eval_metric_st = np.append(eval_metric_st,eval_metric)           
            np.savetxt('eval_metric_per_step_run_'+str(i_run)+'.txt',eval_metric_st, newline="\n")

            # Storing reward after every step
            reward_st_per_step = np.append(reward_st_per_step,reward)
            np.savetxt('reward_per_step_run_'+str(i_run)+'.txt',reward_st_per_step, newline="\n")

            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,False)
            counter+=1
                
            #train critic and actor network
            if counter > start_training: 
                    agent.train()
            print ('\n\n')
             
            reward_per_step=reward
            reward_per_episode+=reward_per_step  

        #After episode ends:
        print ('Episode: ',episode,' Episode Reward: ',reward_per_episode)
        print ("Printing reward to file")
        exploration_noise.reset() #reinitializing random noise for action exploration
        reward_st_per_episode = np.append(reward_st_per_episode,reward_per_episode)
        np.savetxt('episode_reward_run_'+str(i_run)+'.txt',reward_st_per_episode, fmt='%f', newline="\n")
        print ('\n\n')
                     
        total_reward+=reward_per_episode  

        if (best_reward<reward_per_episode):
            best_reward=reward_per_episode
            print('best reward:',best_reward)
            print('current reward:',reward_per_episode)
            print('saving policy for episode..................:',episode)
            agent.save_actor(episode,i_run)

        # Printing eval_metric after every step
        eval_metric=np.array(env.get_eval())
        eval_metric=eval_metric.reshape(-1)
        print('Distance to goal at the end  of episode:',eval_metric)    
        eval_metric_st_per_episode = np.append(eval_metric_st_per_episode,eval_metric)           
        np.savetxt('eval_metric_per_epispde_run_'+str(i_run)+'.txt', eval_metric_st_per_episode, newline="\n")

    print ("Average reward per episode {}".format(total_reward / num_episodes))
    print('Best episode reward',best_reward)   

    del agent
    del frame_obj

if __name__=='__main__':
    from datetime import datetime
    start_time=str(datetime.now())
    run_start=int(sys.argv[1])
    run_end=int(sys.argv[2])
    print('Start trial:',run_start,'End trail:',run_end-1)
    for i_run in range(run_start,run_end):
        s2l(i_run)
    print('Start to end time:',start_time,str(datetime.now()))
