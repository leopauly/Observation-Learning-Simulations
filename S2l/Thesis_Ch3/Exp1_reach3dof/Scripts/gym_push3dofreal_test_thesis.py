#### Training agent in Pusher7Dof gym env using a single real-world env
## Wrtitten by : leopauly | cnlp@leeds.ac.uk
## To be used : as of 05/05/2018
## Courtesy for DDPG implementation : Steven Spielberg Pon Kumar (github.com/stevenpjg)
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
from Policy import Saved_Policy

## Defining env
env = gym.make('Pusher3DOFReal-v1')
env.switch=int(sys.argv[1])
sav_model_num=sys.argv[2]
env.initialize_env()
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"

## Defining vars for reinfrocement learning algo
num_episodes=1
num_rollouts=1 # Each roll out represent a complete activity : activity could be pushing an object, reaching to a point or similar !
steps=60 # No of actions taken in a roll out
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
feature_size=4608 

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


### DEMO FEATURE EXTRACTION
def get_compress_frames_data(filename, num_frames_per_clip=cluster_length):
  ret_arr = []
  for parent, dirnames, filenames in os.walk(filename):

    filenames = sorted(filenames)
    jump=math.floor((len(filenames)/num_frames_per_clip))
    loop=0

    for i in range(0,len(filenames),jump):
      if (loop>15):
        break
      if (filenames[i].endswith('.png')):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
        loop=loop+1
  ret_arr=np.array(ret_arr)
  #ret_arr=ret_arr/255

  return ret_arr

def demo_array_extractor(demo_vid_path):
    demo_vid_array=get_compress_frames_data(demo_vid_path)
    return demo_vid_array
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def distance(f_demo,f_robo):
    #print('shape f_demo',f_demo.shape,'shape f_demo',f_robo.shape)
    return np.linalg.norm(f_demo-f_robo)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


def s2l():

    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    num_states = feature_size   #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]   
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)

    agent = Saved_Policy(num_states,num_actions,sav_model_num)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0 
    total_reward=0
    
    print ("Number of Rollouts per episode:", num_rollouts)
    print ("Number of Steps per roll out:", steps)
    reward_st = np.array([0])  #saving reward
    reward_st_all = np.array([0])  #saving reward after every step
    eval_metric_st= np.array([0]) #saving evalutation metric
    
   
    frame_obj=Frame_Feature()

    for episode in range(num_episodes):
        print ("==== Starting episode no:",episode,"====","\n")
        env.reset()   # Reset env in the begining of each episode
        env.render()
        obs_img=env.render(mode='rgb_array')   # Get the observation
        obs_img=np.array(misc.imresize(obs_img,[112,112,3]))
        observation =np.array(frame_obj.frame_feature_extractor(obs_img))
        observation=observation.reshape(-1)
        reward_per_episode = 0

        for t in range(num_rollouts):  
        
            reward_per_rollout=0
            vid_robo_=[]

            for i in range(steps):

                x = observation

                # Printing eval_metric after every rollout
                eval_metric=np.array(env.get_eval())
                eval_metric=eval_metric.reshape(-1)
                print('Distance to goal:',eval_metric)    
                eval_metric_st = np.append(eval_metric_st,eval_metric)           
                np.savetxt('test_eval_metric_per_step'+str(sav_model_num)+'.txt',eval_metric_st, newline="\n")

                action = agent.get_action(np.reshape(x,[1,num_states]))
                action = action+1
                noise = exploration_noise.noise()
                #action = action[0] + noise*100 #Select action according to current policy and exploration noise
                print ('Action at episode-',episode,'rollout-',t, 'step-', i ," :",action)

               
                _,_,done,info=env.step(action)
                env.render()
                obs_robo_=env.render(mode='rgb_array')   # Get the observation
                obs_robo=misc.imresize(obs_robo_,[112,112,3])
                vid_robo_.append(obs_robo)
                observation=np.array(frame_obj.frame_feature_extractor(np.array(obs_robo)))
                observation=observation.reshape(-1)
                #pasue()
    del agent
    del frame_obj

                
        
            
        
s2l()

