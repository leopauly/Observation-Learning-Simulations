#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt


y = np.loadtxt("test_eval_metric_per_rollout_normalise.txt", unpack=True)

y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
#print(x,y_new)

plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('rollouts')
plt.ylabel('Eval metric per rollout')
plt.show()

y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))

print(x,y_norm)	
plt.figure(2)
plt.plot(x,y_norm)
plt.title('Eval metric Normalised')
plt.xlabel('rollouts')
plt.ylabel('Eval metric per rollout')
plt.show()
