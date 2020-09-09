## To test different gym environmnets

import gym
import imageio
import numpy as np
import sys
import os

os.system('source activate rllab3')

env = gym.make('Pusher3DOFReal-v1')
env.switch=int(sys.argv[1])
env.initialize_env()
env.reset()


env_image=np.array(env.render(mode='rgb_array'))
env.render()
imageio.imwrite("./env_img.png",env_image)
input('press')


