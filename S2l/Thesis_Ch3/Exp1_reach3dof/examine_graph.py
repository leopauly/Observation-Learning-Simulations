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
network_name='activity_model.ckpt-67.meta'
 
## Activity_net n' ## Restore model weights from previously saved model
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
saver = tf.train.import_meta_graph(os.path.join( saved_path, network_name))
saver.restore( sess, os.path.join( saved_path,'activity_model.ckpt-67'))
print("Model restored from file: %s" %  saved_path,flush=True)    


#print(tf.contrib.graph_editor.get_tensors( g))   #(tf.get_default_graph()))
#print(tf.get_default_graph().as_graph_def())
#print([n.name for n in   g.as_graph_def().node])
#print(tf.contrib.graph_editor.get_tensors(g))

writer=tf.summary.FileWriter("./logdir",sess.graph)
writer.close()

print("===========End=================")

#graph_view=tfg.board(tf.get_default_graph())
#graph_view.view()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


