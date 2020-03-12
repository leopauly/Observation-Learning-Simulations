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
from ou_noise import OUNoise
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
start_training=64 # Buffer size, before starting to train the RL algorithm
height=112 
width=112 
channel=3
crop_size=112
cluster_length=16 # No: of frames in a video
nb_classes=2 
feature_size=4608 # state feature size 
demo_folder='./Demos/Task1_0deg_obj/'
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Frame_Feature:
    def __init__(self):
         self.g=tf.Graph()
         with self.g.as_default():
              self.sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
              self.base_model=tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(height,width,channel), pooling=None, classes=1000)
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

def sampling_obs(vid_robo_all,num_frames_per_clip=cluster_length):
    total_obs=len(vid_robo_all)
    jump=math.floor(total_obs/num_frames_per_clip)
    print('jump',jump,'totol ob',total_obs,'vid_robo_all',vid_robo_all.shape)
    loop=0
    ret_arr=[]
    for i in range(0,total_obs,jump):
        if (loop>15):
            break
        img_data = vid_robo_all[i]
        ret_arr.append(img_data)
        loop=loop+1
        
    ret_arr=np.array(ret_arr)
    print('demo array size:::::::',ret_arr.shape)
    #ret_arr=ret_arr/255
    return ret_arr

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

### VIDEO FEATURE EXTRACTION

class Vid_Feature:
    
    def __init__(self):
        self.saved_path='/home/ironman2/S2l_storage/trained_activity_nets_thesis/' 
        self.network_name='activity_model.ckpt-67.meta'
        self.network_weight_name='activity_model.ckpt-67'
        ### Activity_net
        self.g=tf.Graph()
        with self.g.as_default():

            self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            ## Restore model weights from previously saved model
            self.saver = tf.train.import_meta_graph(os.path.join(self.saved_path,self.network_name))
            self.saver.restore(self.sess, os.path.join(self.saved_path,self.network_weight_name))
            print("Model restored from file: %s" % self.saved_path,flush=True)    

    ## For extracting activity features
    def feature_extractor(self,vid_np):
        self.vid_=vid_np.reshape(-1,cluster_length,height,width,channel)
        
        f_v = self.sess.graph.get_tensor_by_name('flatten_1/Reshape:0') #('fc6/Relu:0')
        self.f_v_val=np.array(self.sess.run([f_v], feed_dict={'conv1_input:0':self.vid_,'Placeholder:0':self.vid_ }))

        self.features=np.reshape(self.f_v_val,(-1))
        return self.features

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def distance(f_demo,f_robo):
    return np.linalg.norm(f_demo-f_robo)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

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
    
    activity_obj=Vid_Feature()
    demo_vid_array=demo_array_extractor(demo_folder)
    demo_features=activity_obj.feature_extractor(demo_vid_array)

    frame_obj=Frame_Feature()

    for episode in range(num_episodes):
        print ("==== Starting episode no:",episode,"====","\n")
        observation=env.reset()   
        env.render()
        obs_img=env.render(mode='rgb_array')   
        obs_img=np.array(misc.imresize(obs_img,[112,112,3]))
        #observation =np.array(frame_obj.frame_feature_extractor(obs_img))
        #observation=observation.reshape(-1)
        reward_per_episode = 0

        vid_robo_=[]

        for i in range(steps):

            x = observation

            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()/((episode+1)*1.5)
            print('noise',noise)
            action = action[0] + noise 
            print ('Action at episode-',episode, 'step-', i ," :",action)

            action_max = np.array(env.action_space.high).tolist()
            action_min = np.array(env.action_space.low).tolist()
            action=np.clip(action,action_min,action_max)
            print ('Action at episode clipped-',episode, 'step-', i ," :",action)
               
            observation,get_eval,done,info=env.step(action)
            env.render()
            obs_robo_=env.render(mode='rgb_array')   # Get the observation
            obs_robo=misc.imresize(obs_robo_,[112,112,3])
            vid_robo_.append(obs_robo)
            #observation=np.array(frame_obj.frame_feature_extractor(np.array(obs_robo)))
            #observation=observation.reshape(-1)
                
            if(i==(steps-1)):
                vid_robo_all=np.array(vid_robo_)
                vid_robo=sampling_obs(vid_robo_all)
                robo_features=activity_obj.feature_extractor(vid_robo)
                reward=-(distance(demo_features,robo_features))
                reward=np.array(reward)
                print('reward: ',reward)
            else:
                reward=0
                reward=np.array(reward)
                print('reward: ',reward)

            ## Printing eval_metric after every step
            eval_metric=np.array(custom_get_eval(observation))
            eval_metric=eval_metric.reshape(-1)
            print('Distance to goal:',eval_metric)    
            eval_metric_st = np.append(eval_metric_st,eval_metric)           
            np.savetxt('eval_metric_per_step_run_'+str(i_run)+'.txt',eval_metric_st, newline="\n")

            ## Storing reward after every step
            reward_st_per_step = np.append(reward_st_per_step,reward)
            np.savetxt('reward_per_step_run_'+str(i_run)+'.txt',reward_st_per_step, newline="\n")

            ## add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,False)
            counter+=1
                
            #train critic and actor network
            if counter > start_training: 
                    agent.train()
            print ('\n\n')
            
            reward_per_step=reward
            reward_per_episode+=reward_per_step  

        #check if episode ends:
        
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
        eval_metric=np.array(custom_get_eval(observation))
        eval_metric=eval_metric.reshape(-1)
        print('Distance to goal at the end  of episode:',eval_metric)    
        eval_metric_st_per_episode = np.append(eval_metric_st_per_episode,eval_metric)           
        np.savetxt('eval_metric_per_epispde_run_'+str(i_run)+'.txt', eval_metric_st_per_episode, newline="\n")

    print ("Average reward per episode {}".format(total_reward / num_episodes))
    print('Best episode reward',best_reward)   

    del agent
    del activity_obj
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
