## To test different gym environmnets

import gym
import imageio
import numpy as np

env = gym.make('Pusher3DOFReal-v1')
env.switch=0
env.initialize_env()
env.reset()
input('Press key')

print((env.action_space.high))
print((env.action_space.low))

while True: 
	env_image=np.array(env.render(mode='rgb_array'))
	env.render()
	print(env_image.shape)
	imageio.imwrite("./env_img.png",env_image)
	input('Press key')




