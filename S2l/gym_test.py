## To test different gym environmnets

import gym
env = gym.make('Pusher7DOF-v1')
env.reset()
while True: 
	env.render()
