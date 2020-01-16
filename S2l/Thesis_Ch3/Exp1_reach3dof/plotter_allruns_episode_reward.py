#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

plt.ion()
num_runs=int(sys.argv[1])

while(True):
	for run in range(num_runs):
			try:
				y = np.loadtxt('reward_per_step_run_'+str(run)+'.txt', unpack=True)
				y_new=[y_ for y_ in y if y_!=0]
				x=range(len(y_new))
				#print(x,y_new)

				plt.figure(1)
				plt.plot(x,y_new)
				plt.title('Reward')
				plt.xlabel('Episode')
				plt.ylabel('reward per episode')
				#plt.savefig('Rewards.png')
				plt.show()
				plt.pause(1)
			except:
				continue


