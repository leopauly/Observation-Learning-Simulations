#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import sys

run=sys.argv[1]

y = np.loadtxt('eval_metric_per_step_run_'+run+'.txt', unpack=True)

y_new=y[1:len(y)]
x=range(len(y_new))
print(x,y_new)

plt.figure(1)
plt.plot(x,y_new)
plt.title('Eval metric')
plt.xlabel('steps')
plt.ylabel('Eval metric per steps')
plt.show()

