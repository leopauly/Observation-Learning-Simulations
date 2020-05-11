## To test different gym environmnets

import gym
import imageio
import numpy as np

env = gym.make('Pusher3DOFReal-v1')
env.switch=3
env.initialize_env()
env.reset()

while True: 
	env_image=np.array(env.render(mode='rgb_array'))
	env.render()
	print(env_image.shape)
	imageio.imwrite("./env_img.png",env_image)
	input('Press key')




