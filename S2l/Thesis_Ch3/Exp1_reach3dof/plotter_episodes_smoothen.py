#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import sys


run=sys.argv[1]

window_size=int(sys.argv[2])
y = np.loadtxt('episode_reward_run_'+run+'.txt', unpack=True)
y_new=y[1:len(y)]

x=range(len(y_new))
plt.figure(1)
plt.plot(x,y_new)
plt.title('Original Reward')
plt.xlabel('episodes')
plt.ylabel('reward per episeodes')
plt.show()

y_new_smooth=[]
for i in range(window_size,len(y_new)-window_size):
	k=0
	k=y_new[i]		
	for j in range (1,window_size+1):
		k=k+y_new[i-j]+y_new[i+j]
	y_new_smooth.append(k/((2*window_size)+1))
	#y_new_smooth.append((y_new[i-1]+y_new[i-2]+y_new[i-3]+y_new[i-4]+y_new[i]+y_new[i+1]+y_new[i+2]+y_new[i+3]+y_new[i+4])/window_size)
   
x_smooth=range(len(y_new_smooth))
plt.figure(2)
plt.plot(x_smooth,y_new_smooth)
plt.title('Smoothened reward')
plt.xlabel('episodes')
plt.ylabel('Smoothened reward per episeodes')
plt.show()

