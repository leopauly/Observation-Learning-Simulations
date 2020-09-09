## To test different gym environmnets

import gym
env = gym.make('MultiViewPusher-v0')
env.reset()
while True: 
	env.render()
