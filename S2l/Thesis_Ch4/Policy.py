#### 
## written by :  Leo Pauly | @ leopauly
## To load the saved policy network
####

import tensorflow as tf
import numpy as np
import os
import math

class Saved_Policy:
    
    def __init__(self,num_states,num_actions):
        self.policy_savedpath="/home/ironman2/S2l_storage/policies_saved/Thesis_Ch4/Task1/V_new"    
        self.network_name='episode-298'
        self.g=tf.Graph()

        ## Restore model weights from previously saved model
        with self.g.as_default():
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            _,_,_,_,_,_,self.actor_state_in,self.actor_model = self.create_actor_net(num_states, num_actions)
            #self.init_op=tf.initialize_all_variables()
            #self.sess.run(self.init_op)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, os.path.join(self.policy_savedpath,self.network_name))
            print("Model restored from file: %s" % self.policy_savedpath,flush=True)    



    ## Network that takes states and return action
    def create_actor_net(self, num_states=4, num_actions=1):
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        actor_state_in = tf.placeholder("float",[None,num_states])    
        W1_a=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1_a=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2_a=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        B2_a=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        W3_a=tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-0.003,0.003))
        B3_a=tf.Variable(tf.random_uniform([num_actions],-0.003,0.003))
    
        H1_a=tf.nn.softplus(tf.matmul(actor_state_in,W1_a)+B1_a)
        H2_a=tf.nn.tanh(tf.matmul(H1_a,W2_a)+B2_a)
        actor_model=tf.matmul(H2_a,W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, actor_state_in, actor_model

    ## Getting actions
    def get_action(self,state_t):
        self.action=np.array(self.sess.run(self.actor_model, feed_dict={self.actor_state_in:state_t}))
        return self.action  
        
        
        
