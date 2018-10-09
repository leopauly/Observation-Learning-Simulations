#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt

window_size=5
y = np.loadtxt("episode_reward.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]

x=range(len(y_new))
plt.figure(1)
plt.plot(x,y_new)
plt.title('Original Reward')
plt.xlabel('episodes')
plt.ylabel('reward per episeodes')
plt.show()

y_new_smooth=list()
for i in range(window_size,len(y_new)-window_size):
	y_new_smooth.append((y_new[i-1]+y_new[i-2]+y_new[i-3]+y_new[i-4]+y_new[i]+y_new[i+1]+y_new[i+2]+y_new[i+3]+y_new[i+4])/window_size)
   
x_smooth=range(len(y_new_smooth))
plt.figure(2)
plt.plot(x_smooth,y_new_smooth)
plt.title('Smoothened Reward')
plt.xlabel('episodes')
plt.ylabel('reward per episeodes')
plt.show()

