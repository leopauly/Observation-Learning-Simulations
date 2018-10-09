## @leopauly
## For finding correlation between rewards per rollout vs eval_metric per rollout


## loading values
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
window_size=2

#-------------------------------------------------------------------------------------------------#

y = np.loadtxt("reward_all.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]
print('y size:' ,np.array(y_new).shape)
#print('y :' ,y_new)

y_new_smooth=list()
y_new_smooth.append(0)

for i in range(window_size-1,len(y_new)-window_size+1):
  
    y_new_smooth.append((y_new[i-1]+y_new[i]+y_new[i+1])/window_size)
y_new_smooth.append(0)

#-------------------------------------------------------------------------------------------------#

x = np.loadtxt("eval_metric_per_step.txt", unpack=True)
x_new=[x_ for x_ in x if x_!=0]
#print(x_new)
x_new_per_roll=[]
for i in range(15,len(x_new),16):
    #print(len(x_new),i)
    x_new_per_roll.append(x_new[i])

x_norm=[]
for i in range(len(x_new_per_roll)):
	x_norm.append(1-(x_new_per_roll[i]/x_new_per_roll[0]))	

print('x size:' ,np.array(x_new_per_roll).shape)
#print('x :' ,x_new_per_roll)

#-------------------------------------------------------------------------------------------------

## plotting correlation coefficient for the paper
cor_coef=np.corrcoef(x_norm,y_new)
print(np.corrcoef(x_norm,y_new))

plt.ylabel('Perceptual Reward')
plt.xlabel('Auxiliary Reward')
plt.scatter(x_norm,y_new)
plt.savefig('Perceptual Reward vs Auxiliary Reward')
plt.show()


