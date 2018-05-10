#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.ion()

while (True):
	y = np.loadtxt("reward_all.txt", unpack=True)
	y_new=[y_ for y_ in y if y_!=0]
	x=range(len(y_new))
	#print(x,y_new)

	plt.figure(1)
	plt.plot(x,y_new)
	plt.title('Reward')
	plt.xlabel('rollouts')
	plt.ylabel('reward per rollout')
	plt.show()
	plt.pause(3)


