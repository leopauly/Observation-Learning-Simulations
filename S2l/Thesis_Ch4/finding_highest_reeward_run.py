#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

plt.ion()
num_runs=int(sys.argv[1])

max_per_run=[]
episode_max_reward_per_run=[]
for run in range(num_runs):
	try:
		y = np.loadtxt('reward_per_step_run_'+str(run)+'.txt', unpack=True)
		y_new=[y_ for y_ in y if y_!=0]
		max_per_run.append(np.max(y_new))
		episode_max_reward_per_run.append(np.argmax(y_new))
	except:
		continue
print('Heightest reward per run',max_per_run)
print('Episode index of heightest reward per run',episode_max_reward_per_run)
ranking=np.argsort(np.array(max_per_run))

print('---------------------------------------------------------',)
print('Runs sorted in the heigherst reward order',ranking[::-1])
ranking_high_low=ranking[::-1]
best_episode_best_run=[]
for i in range(num_runs):
	best_episode_best_run.append(episode_max_reward_per_run[ranking_high_low[i]])
print('Best episodes ordered according to best runs',best_episode_best_run)
