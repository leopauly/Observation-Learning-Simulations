#### Training agent in Pusher7Dof gym env using a single real-world env
## Wrtitten by : leopauly | cnlp@leeds.ac.uk

##Imports
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc as misc

## Imports for DNN
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

## Custom scripts
import lscript as lsp
import modelling as md

## Defining env
env = gym.make('Pusher7DOF-v1')
env.reset()

## Defining vars for reinfrocement learning algo
num_episodes=2
num_rollouts=10
LR = 1e-3
goal_steps = 500
score_requirement = 50
initial_games = 10000

## vars for feature extraction
height=112 
width=112 
channel=3
crop_size=112

cluster_length=16
nb_classes=2
feature_size=4608 #8192   #16384  #487
saved_path='/home/ironman/trained_activity_nets/'
batch_size=32
demo_folder='./Demo_reach_1/'

## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel],name='x') 
y_true = tf.placeholder(tf.float32, [None, nb_classes],name='y_true')
y_true_cls = tf.placeholder(tf.int64, [None],name='y_true_cls')

model_keras = md.C3D_ucf101_training_model_tf(summary=True)
out=model_keras(x_image)
y_pred = tf.nn.softmax(out)
y_pred_cls = tf.argmax(out, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Loading netwrok framework finished..!!',flush=True)

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

## Start the session with logging placement.
init_op = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init_op)


## Restore model weights from previously saved model
saver = tf.train.import_meta_graph(os.path.join(saved_path,'activity_model.ckpt-104.meta'))
saver.restore(sess, os.path.join(saved_path,'activity_model.ckpt-104'))
print("Model restored from file: %s" % saved_path,flush=True)

def demo_feature_extractor(demo_vid_path):
    demo_vid_array=get_compress_frames_data(demo_vid_path)
    return feature_extractor(demo_vid_array)

## For extracting activity features
def feature_extractor(vid_np):
    print('shape of video for feature extraction:',vid_np.shape)
    vid_=vid_np.reshape(-1,cluster_length,height,width,channel)
            
    #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
    #print(tf.get_default_graph().as_graph_def())
    f_v = sess.graph.get_tensor_by_name('flatten_1/Reshape:0')
    f_v_val=np.array(sess.run([f_v], feed_dict={'conv1_input:0':vid_,x_image:vid_,K.learning_phase(): 0 }))

    print('extracted video features shape:',f_v_val.shape)
    features=np.reshape(f_v_val,(-1))
    print('features_shape',features.shape)
    return features

def distance(f_demo,f_robo):
    print('shape f_demo',f_demo.shape,'shape f_demo',f_robo.shape)
    return np.linalg.norm(f_demo-f_robo)


def s2l():

    demo_features=demo_feature_extractor(demo_folder)
    for episode in range(num_episodes):
        env.reset()   # Reset env in the begining of each episode
        for t in range(num_rollouts):  
           
            #env.render()  # Obtaining visual state
            #action = env.action_space.sample() # This will just create a sample action in any environment.
            #print(action)
            
            action=[1,1,1,1,1,1,1] 
            _,_, done, info = env.step(action)  # this executes the environment with an action,and returns the observation of the environment, the reward, if the env is over, and other info.

            
            vid_robo_=[]
            for i in range(16):
                obs=env.render(mode='rgb_array')   # Get the observation
                obs_new=misc.imresize(obs,[112,112,3])
                vid_robo_.append(obs_new)
            vid_robo=np.array(vid_robo_)
            robo_features=feature_extractor(vid_robo)


            reward=-(distance(demo_features,robo_features))
            print('reward',reward)

            #obs_features=model(obs) # Convert observation into features
            #print(obs.shape)
            #plt.imshow(obs)
            #plt.show()

            #cv2.imshow('Sim',obs)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            '''
            obs_features=model(obs) # Convert observation into features
            ## action=policy_fn(obs_features,)
            _,_, done, info = env.step(action) 

            obs=env.render()   # Get the observation
            obs_features=model(obs) # Convert observation into features
            reward=reward_fn(1-1)
            ## update_policy(reward)
        
            #if done:
            #    break
            '''



s2l()

