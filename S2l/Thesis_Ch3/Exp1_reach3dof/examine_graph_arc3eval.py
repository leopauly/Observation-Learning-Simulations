#### 
## Wrtitten by : leopauly | cnlp@leeds.ac.uk
## Exp for viewing a tensoflow graph
####

## Imports
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import skimage
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.color import grey2rgb,rgb2grey
from skimage.feature import hog
from skimage import io

# Custom scripts
import lscript as lsp
import modelling as md

## Imports for DNN
import tensorflow as tf
from keras import backend as K


height=112 
width=112 
channel=3
crop_size=112
cluster_length=16
feature_size=8192 #4096 #16384
baseline_feature_size=4608
baseline2_feature_size=2


nb_activities=4
nb_videos=5
nb_pixels_per_cell=(8,8)
nb_cells_per_block=(1,1)
nb_orientations=16
saved_path='/home/ironman2/S2l_storage/trained_activity_nets_thesis/' 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


## Defining placeholders in tf for images and targets
x_image = tf.placeholder(tf.float32, [None, 16,height,width,channel],name='x') 

model_keras = md.C3D_ucf101_training_model_tf(summary=True)
out=model_keras(x_image)

#### Start the session with logging placement.
init_op=tf.global_variables_initializer()
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init_op)

### Restore model weights from previously saved model
saver=tf.train.Saver() #saver=tf.train.import_meta_graph(os.path.join(saved_path, 'activity_model.ckpt-67.meta'))
saver.restore(sess, os.path.join(saved_path,'activity_model.ckpt-67'))
print("Model restored from file: %s" % saved_path,flush=True)




## Examining Graph
writer=tf.summary.FileWriter("./logdir_arc3eval",sess.graph)
writer.close()

print("===========End=================")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


