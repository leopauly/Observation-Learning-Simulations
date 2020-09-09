#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt


y = np.loadtxt("test_eval_metric_per_rollout.txt", unpack=True)

y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
print(x,y_new)

plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('rollouts')
plt.ylabel('Eval metric per rollout')
plt.show()

