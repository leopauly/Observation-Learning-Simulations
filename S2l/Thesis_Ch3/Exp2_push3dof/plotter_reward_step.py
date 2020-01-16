#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import sys

run=sys.argv[1]

y = np.loadtxt('reward_per_step_run_'+run+'.txt', unpack=True)

y_new=y# [y_ for y_ in y if y_!=0]
x=range(len(y_new))
print(x,y_new)

plt.figure(1)
plt.plot(x,y_new)
plt.title('Reward')
plt.xlabel('steps')
plt.ylabel('reward per step')
plt.show()

y_new=-np.array(y_new)

plt.figure(2)
plt.plot(x,y_new)
plt.title('Feature distance')
plt.xlabel('rollouts')
plt.ylabel('reward per rollout')
plt.show()
