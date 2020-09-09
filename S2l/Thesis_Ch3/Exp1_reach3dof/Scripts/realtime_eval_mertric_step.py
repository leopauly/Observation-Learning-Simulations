#### For plotting from reawrd values stored in files


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

plt.ion()
run=sys.argv[1]
while (True):
	y = np.loadtxt('eval_metric_per_step_run_'+run+'.txt', unpack=True)
	y_new=y[1:len(y)]
	x=range(len(y_new))


	plt.figure(1)
	plt.plot(x,y_new)
	plt.title('Evaluation Metric')
	plt.xlabel('Step')
	plt.ylabel('Distance per step')
	plt.show()
	plt.pause(3)


