#### 
## written by :  Leo Pauly | @ leopauly
## To load the saved policy network
####


class Saved_Policy:
    
    def __init__(self):
        self.policy_savepath="/home/ironman/S2l_Storage/policies_saved/Exp2/"
        self.network_graph_name='policy.episode-0.meta'
        self.network_name='activity_model.ckpt-104'
        ### policy_net
        self.g=tf.Graph()
        with self.g.as_default():

            self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            ## Restore model weights from previously saved model
            self.saver = tf.train.import_meta_graph(os.path.join(self.saved_path,self.network_graph_name))
            self.saver.restore(self.sess, os.path.join(saved_path,self.network_name))
            print("Model restored from file: %s" % policy_savepath,flush=True)    

    ## For extracting activity features
    def get_action(self,state_t):
        print(tf.contrib.graph_editor.get_tensors(self.g))   #(tf.get_default_graph()))
        print(tf.get_default_graph().as_graph_def())
        action_tensor = self.sess.graph.get_tensor_by_name('flatten_1/Reshape:0')
        self.action=np.array(self.sess.run([action_tensor], feed_dict={self.actor_state_in:state_t}))#,K.learning_phase(): 0 }))
        print (self.action.shape)

        return self.action.shape # self.sess.run(self.actor_model, feed_dict={self.actor_state_in:state_t})  
        
        
        '''
        #print('shape of video for feature extraction:',vid_np.shape)
        self.vid_=vid_np.reshape(-1,cluster_length,height,width,channel)

        #print(tf.contrib.graph_editor.get_tensors(self.g))   #(tf.get_default_graph()))
        #print(tf.get_default_graph().as_graph_def())
        f_v = self.sess.graph.get_tensor_by_name('flatten_1/Reshape:0')
        self.f_v_val=np.array(self.sess.run([f_v], feed_dict={'conv1_input:0':self.vid_,'Placeholder:0':self.vid_ }))#,K.learning_phase(): 0 }))

        #print('extracted video features shape:',f_v_val.shape)
        self.features=np.reshape(self.f_v_val,(-1))
        #print('features_shape',features.shape)
        return self.features