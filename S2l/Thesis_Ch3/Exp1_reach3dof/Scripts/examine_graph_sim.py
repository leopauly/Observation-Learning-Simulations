#### 
## Wrtitten by : leopauly | cnlp@leeds.ac.uk
## Exp for viewing a tensoflow graph
####

## Imports
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
import tfgraphviz as tfg

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


## Examining Graph

saved_path='/home/ironman2/S2l_storage/trained_activity_nets_thesis/' 
network_graph_name='activity_model.ckpt-67.meta'
network_weight_name='activity_model.ckpt-67'
 
## Activity_net n' ## Restore model weights from previously saved model
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
saver = tf.train.import_meta_graph(os.path.join( saved_path,network_graph_name))
saver.restore( sess, os.path.join( saved_path,network_weight_name))


writer=tf.summary.FileWriter("./logdir_sim",sess.graph)
writer.close()

print("===========End=================")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


