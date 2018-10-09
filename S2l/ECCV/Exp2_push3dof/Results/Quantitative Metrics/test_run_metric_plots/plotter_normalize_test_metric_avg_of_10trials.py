#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
rollout_per_episode=20
num_trails=10
import matplotlib
matplotlib.rcParams.update({'font.size':14})

#0 degree
y = np.loadtxt("0deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(rollout_per_episode)
print(y_new)


y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))	

y_norm_avg=[]
for i in range(rollout_per_episode):
	sum=0
	for j in range(num_trails):
		sum=sum+y_norm[i+(j*20)]
	y_norm_avg.append(sum/num_trails)


plt.figure(2)
plt.plot(x,y_norm_avg,label='View-1') #label='$0^{0}$')
#plt.hlines(1,-1,rollout_per_episode,colors='k',linestyles='dashed')
#red_patch = mpatches.Patch(color='organge', label='')
#plt.legend(handles=[red_patch])



#plt.title('Eval metric Normalised')
plt.xlabel('Rollouts')
plt.ylabel('Task completion rate')

'''
#45 degree
y = np.loadtxt("45deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]

y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))	

y_norm_avg=[]
for i in range(rollout_per_episode):
	sum=0
	for j in range(num_trails):
		sum=sum+y_norm[i+(j*20)]
	y_norm_avg.append(sum/num_trails)

plt.figure(2)
plt.plot(x,y_norm_avg,label='$45^{0}$')
'''


#90 degree
y = np.loadtxt("90deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]


y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))

y_norm_avg=[]
for i in range(rollout_per_episode):
	sum=0
	for j in range(num_trails):
		sum=sum+y_norm[i+(j*20)]
	y_norm_avg.append(sum/num_trails)


plt.figure(2)
plt.plot(x,y_norm_avg,label='View-2') #label='$90^{0}$')


#180 degree
y = np.loadtxt("180deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]


y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))

y_norm_avg=[]
for i in range(rollout_per_episode):
	sum=0
	for j in range(num_trails):
		sum=sum+y_norm[i+(j*20)]
	y_norm_avg.append(sum/num_trails)


plt.figure(2)
plt.plot(x,y_norm_avg,label= 'View-3') #'$180^{0}$')

ax=plt.gca()

plt.ylim(ymax=1.0,ymin=0.0)

#matplotlib.pyplot.xticks(x)

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
plt.xlim(xmax=20,xmin=0)
#ax.set_xticks=([0,5,10,15,20])
ax.set_xticklabels(['-5','0','5','10','15','20'])
print(x)
plt.legend(loc=4)
plt.show()


