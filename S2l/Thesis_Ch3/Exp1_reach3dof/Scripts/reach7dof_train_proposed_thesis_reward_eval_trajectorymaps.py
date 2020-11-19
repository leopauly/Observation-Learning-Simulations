#### Reward evaluation for agent in Reacher7Dof gym env using a single real-world env
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
import skimage
import os
from threading import Thread, Lock
import sys
from six.moves import xrange  
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

## Defining vars
num_episodes=20
steps=60 # No of actions taken in a roll out
is_batch_norm = False #batch normalization switch
xrange=range # For python3
start_training=64 # Buffer size, before starting to train the RL algorithm
height=112 
width=112 
channel=3
crop_size=112
cluster_length=16 
nb_classes=2 
feature_size=4608 
layer_name=sys.argv[4]


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
## Printing all the experiment  hyper-parameters
print('Switch:',sys.argv[3])
print('Layer name:',sys.argv[4])







#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

switch=int(sys.argv[3])
if(switch==0):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Reward_Eval_Conv5_20eps/rewardmap_misc/'
    demo_folder='../Demos/demo_reach_0deg_new/'
elif (switch==1):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Reward_Eval_Conv5_20eps_rand/rewardmap_3Dview1/'
    demo_folder='../Demos/demo_reach_0deg_new/'
elif (switch==2):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Reward_Eval_Conv5_20eps_rand/rewardmap_3Dview2/'
    demo_folder='../Demos/demo_reach_0deg_new/'
elif (switch==3):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Traj_maps_'+layer_name.split('/')[0]+'_20eps/lr_traj1/'
    demo_folder='../Demos/demo_reach_0deg_new/' 
elif (switch==4):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Traj_maps_'+layer_name.split('/')[0]+'_20eps/ll_traj1/'
    demo_folder='../Demos/demo_reach_0deg_new/' 
elif (switch==5):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Reward_Eval_Conv5_20eps_rand/multi_target_close/'
    demo_folder='../Demos/demo_reach_0deg_new/' 
elif (switch==6):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_Random/Reward_Eval_Conv5_20eps_rand/multi_target_far/'
    demo_folder='../Demos/demo_reach_0deg_new/' 
elif (switch==-2):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/V_new/'
    demo_folder='../Demos/demo_reach_180deg_new/'
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/V_new/'
elif (switch==-4):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/Obj2/'
    demo_folder='../Demos/demo_reach_180deg_new/'  
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/Obj2/'
elif (switch==-3):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/Obj1_new/'
    demo_folder='../Demos/demo_reach_0deg_new/'  
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/Obj1_new/'
elif (switch==-5):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/BG/'
    demo_folder='../Demos/demo_reach_0deg_new/' 
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/BG/'
elif (switch==-6):
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/M/'
    demo_folder='../Demos/demo_reach_0deg_h.s/'
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/M/'
else:
    base_dir='/home/ironman2/Observation-Learning-Simulations/S2l/Thesis_Ch3/Exp1_reach3dof/Results/Results_UCF/Proposed/'+layer_name.split('/')[0]+'/I_new/'
    demo_folder='../Demos/demo_reach_0deg_new/'
    policy_savepath= '/home/ironman2/S2l_storage/policies_saved/thesis/Proposed_'+layer_name.split('/')[0]+'/I_new/'
  
os.system('mkdir %s' % base_dir)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

## Defining env
env = gym.make('Pusher3DOFReal-v1')
env.switch=switch
env.initialize_env()
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"

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

def sampling_obs(vid_robo_all,num_frames_per_clip=cluster_length):
    total_obs=len(vid_robo_all)
    jump=math.floor(total_obs/num_frames_per_clip)
    loop=0
    ret_arr=[]
    for i in range(0,total_obs,jump):
        if (loop>15):
            break
        img_data = vid_robo_all[i]
        ret_arr.append(img_data)
        loop=loop+1
        
    ret_arr=np.array(ret_arr)
    #ret_arr=ret_arr/255
    return ret_arr

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

### VIDEO FEATURE EXTRACTION
class Vid_Feature:
    
    def __init__(self):
        self.saved_path='/home/ironman2/S2l_storage/trained_activity_nets_thesis/saved/models/' 
        self.network_name='activity_model.ckpt-67.meta'
        self.network_weigths_name='activity_model.ckpt-67'

        #self.saved_path='/home/ironman2/S2l_storage/trained_C3D_MIME/' 
        #self.network_name='activity_model.ckpt-155.meta'
        #self.network_weigths_name='activity_model.ckpt-155'
        ### Activity_net
        self.g=tf.Graph()
        with self.g.as_default():

            self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            ## Restore model weights from previously saved model
            self.saver = tf.train.import_meta_graph(os.path.join(self.saved_path,self.network_name))
            self.saver.restore(self.sess, os.path.join(self.saved_path,self.network_weigths_name)) 
            #self.sess.run(tf.global_variables_initializer())
            print("Model restored from file: %s" % self.saved_path,flush=True)    

    ## For extracting activity features
    def feature_extractor(self,vid_np):
        self.vid_=vid_np.reshape(-1,cluster_length,height,width,channel)
        f_v = self.sess.graph.get_tensor_by_name(layer_name) #('flatten_1/Reshape:0')
        self.f_v_val=np.array(self.sess.run([f_v], feed_dict={'conv1_input:0':self.vid_,'Placeholder:0':self.vid_,'dropout_1/keras_learning_phase:0':0 }))
        self.features=np.reshape(self.f_v_val,(-1))
        return self.features

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def distance(f_demo,f_robo):
    norm_val=2
    norm_pow=1
    distance_=np.linalg.norm(f_demo-f_robo,ord=norm_val)
    return pow(distance_,norm_pow)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def s2l(i_run):
    print('This is the ith run',i_run)

    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    num_states = feature_size   #num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]   
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)

    agent = DDPG(env, is_batch_norm,num_states,num_actions,policy_savepath)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0 
    total_reward=0
    best_reward=-10000
    
    print ("Number of Steps per episode:", steps)
    reward_st_per_episode = np.array([0])  #saving reward
    man_pos_x_st_per_episode = np.array([0])
    man_pos_y_st_per_episode = np.array([0])
    eval_metric_st= np.array([0])
    eval_metric_st_per_episode= np.array([0])
    reward_st_per_step = np.array([0])  #saving reward after every step
    
    activity_obj=Vid_Feature()
    demo_vid_array=demo_array_extractor(demo_folder)
    if(i_run==0):
        plt.imshow(demo_vid_array[0])
        plt.savefig('demo_img'+str(switch) +'.png')
    demo_features=activity_obj.feature_extractor(demo_vid_array)
    frame_obj=Frame_Feature()

    for episode in range(num_episodes):
        print ("==== Starting episode no:",episode,"====","\n")
        env.reset()   # Reset env in the begining of each episode
        env.render()
        obs_img=env.render(mode='rgb_array')   # Get the observation
        if(i_run==0 and episode==0):
            plt.imshow(obs_img)
            plt.savefig('env_img'+str(switch) +'.png')
        obs_img=np.array(misc.imresize(obs_img,[112,112,3]))
        observation =np.array(frame_obj.frame_feature_extractor(obs_img))
        observation=observation.reshape(-1)
        
        reward_per_episode = 0
        vid_robo_=[]

        man_pos_x_st_per_step = np.array([0])
        man_pos_y_st_per_step = np.array([0])

        for i in range(steps):

            x = observation

            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            action = action+1
            noise = exploration_noise.noise()#/(episode+1)
            action = action[0] + noise 
            print ('Action at',i_run ,'episode-',episode, 'step-', i ," :",action)

               
            _,_,done,info=env.step(action)
            env.render()
            obs_robo_=env.render(mode='rgb_array')   
            obs_robo=misc.imresize(obs_robo_,[112,112,3])
            vid_robo_.append(obs_robo)
            
            observation=np.array(frame_obj.frame_feature_extractor(np.array(obs_robo)))
            observation=observation.reshape(-1)
                
            if(i==(steps-1)):
                vid_robo_all=np.array(vid_robo_)
                vid_robo=sampling_obs(vid_robo_all)
                #for i in range(len(vid_robo)):
                #    plt.imshow(vid_robo[i])
                #    plt.savefig('obs_img'+str(i) +'.png')
                robo_features=activity_obj.feature_extractor(vid_robo)
                reward=-(distance(demo_features,robo_features))
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
            np.savetxt(base_dir+'eval_metric_per_step_run_'+str(i_run)+'.txt',eval_metric_st, newline="\n")

            ## Printing and saving final mnaipulator position
            manipulator_pos=env.get_man_pos()
            print('Episode: ',episode,'step: ',i,'Manipulator position: ',manipulator_pos)

            man_pos_x_st_per_step = np.append(man_pos_x_st_per_step,manipulator_pos[0])
            np.savetxt(base_dir+'step_man_x_pos_run_'+str(i_run)+'_eps_'+str(episode)+'.txt',man_pos_x_st_per_step, fmt='%f', newline="\n")

            man_pos_y_st_per_step = np.append(man_pos_y_st_per_step,manipulator_pos[1])
            np.savetxt(base_dir+'step_man_y_pos_run_'+str(i_run)+'_eps_'+str(episode)+'.txt',man_pos_y_st_per_step, fmt='%f', newline="\n")
        

            # Storing reward after every step
            reward_st_per_step = np.append(reward_st_per_step,reward)
            np.savetxt(base_dir+'reward_per_step_run_'+str(i_run)+'.txt',reward_st_per_step, newline="\n")

            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,False)
            counter+=1
                
            #train critic and actor network
            if counter > start_training: 
                agent.train() #
            print ('\n\n')
            print('Episode: ',episode,' Manipulator position: ',env.get_man_pos())
              
            reward_per_step=reward
            reward_per_episode+=reward_per_step  

        ## Printing and saving episode rewards
        print ('Episode: ',episode,' Episode Reward: ',reward_per_episode)
        exploration_noise.reset() #reinitializing random noise for action exploration
        reward_st_per_episode = np.append(reward_st_per_episode,reward_per_episode)
        np.savetxt(base_dir+'episode_reward_run_'+str(i_run)+'.txt',reward_st_per_episode, fmt='%f', newline="\n")

        ## Printing and saving final mnaipulator position
        manipulator_final_pos=env.get_man_pos()
        print('Episode: ',episode,' Manipulator position: ',manipulator_final_pos)

        man_pos_x_st_per_episode = np.append(man_pos_x_st_per_episode,manipulator_final_pos[0])
        np.savetxt(base_dir+'episode_man_x_pos_run_'+str(i_run)+'.txt',man_pos_x_st_per_episode, fmt='%f', newline="\n")

        man_pos_y_st_per_episode = np.append(man_pos_y_st_per_episode,manipulator_final_pos[1])
        np.savetxt(base_dir+'episode_man_y_pos_run_'+str(i_run)+'.txt',man_pos_y_st_per_episode, fmt='%f', newline="\n")
        
        ## Saving
        if (best_reward<reward_per_episode):
            best_reward=reward_per_episode
            print('best reward:',best_reward)
            print('current reward:',reward_per_episode)
            print('saving policy for episode..................:',episode)
            agent.save_actor(episode,i_run)
        
        ## Printing eval_metric after every step
        eval_metric=np.array(env.get_eval())
        eval_metric=eval_metric.reshape(-1)
        print('Distance to goal at the end  of episode:',eval_metric)    
        eval_metric_st_per_episode = np.append(eval_metric_st_per_episode,eval_metric)           
        np.savetxt(base_dir+'eval_metric_per_epispde_run_'+str(i_run)+'.txt', eval_metric_st_per_episode, newline="\n")

        total_reward+=reward_per_episode


    print ("Average reward per episode {}".format(total_reward / num_episodes))
    print('Best episode reward',best_reward)   
    print ('\n\n')


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
