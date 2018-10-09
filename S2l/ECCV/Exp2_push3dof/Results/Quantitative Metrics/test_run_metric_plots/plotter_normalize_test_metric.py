#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

#0 degree
y = np.loadtxt("0deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))

y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))	

print(x,y_norm)	
plt.figure(2)
plt.plot(x[0:20],y_norm[0:20])
plt.title('Eval metric Normalised')
plt.xlabel('rollouts')
plt.ylabel('Eval metric per rollout')

#45 degree
y = np.loadtxt("45deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))	
print(x,y_norm)	
plt.figure(2)
plt.plot(x[0:20],y_norm[0:20])


#90 degree
y = np.loadtxt("90deg.txt", unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
y_norm=[]
for i in range(len(y_new)):
	y_norm.append(1-(y_new[i]/y_new[0]))
print(x,y_norm)	
plt.figure(2)
plt.plot(x[0:20],y_norm[0:20])

plt.show()
