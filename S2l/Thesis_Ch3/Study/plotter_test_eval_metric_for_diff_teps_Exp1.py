#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import sys


y = np.loadtxt('./Exp1/test_eval_metric_per_step_Study.Exp1.60ps.txt', unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
print(x,y_new)
plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('steps')
plt.ylabel('Eval metric per step')

y = np.loadtxt('./Exp1/test_eval_metric_per_step_Study.Exp1.110eps.txt', unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
print(x,y_new)
plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('steps')
plt.ylabel('Eval metric per step')

y = np.loadtxt('./Exp1/test_eval_metric_per_step_Study.Exp1.160eps.txt', unpack=True)
y_new=[y_ for y_ in y if y_!=0]
x=range(len(y_new))
print(x,y_new)
plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('steps')
plt.ylabel('Eval metric per step')

plt.savefig('Study_steps_Exp1')
plt.show()

