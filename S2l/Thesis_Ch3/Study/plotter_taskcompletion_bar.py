#### For plotting task completion rates in bar plot for simulation experiments


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def read_file(filenames):
	y = np.loadtxt(filenames, unpack=True)
	y_new=[y_ for y_ in y if y_!=0]
	y_norm=[]
	for i in range(len(y_new)):
		y_norm.append(1-(y_new[i]/y_new[0]))	

	return y_norm


def mean_std(filenames):
	trials=[]
	for i in range(len(filenames)):
		y_norm=read_file(filenames[i])
		trials.append(y_norm[len(y_norm)-1])
			
	trails=np.array(trials)
	mean=np.mean(trails)
	std=np.std(trails)
	return mean,std


#------------------------------------------------------------------------------------------------------------#	



filenames0_p=['./Exp1/test_eval_metric_per_step_Study.Exp1.60ps.txt','./Exp1/test_eval_metric_per_step_Study.Exp1.60ps.txt']
filenames1_p=['./Exp1/test_eval_metric_per_step_Study.Exp1.110eps.txt','./Exp1/test_eval_metric_per_step_Study.Exp1.110eps.txt']
filenames2_p=['./Exp1/test_eval_metric_per_step_Study.Exp1.160eps.txt','./Exp1/test_eval_metric_per_step_Study.Exp1.160eps.txt']

filenames0_b=['./Exp2/test_eval_metric_per_step_Study.Exp2.60ps.txt','./Exp2/test_eval_metric_per_step_Study.Exp2.60ps.txt']
filenames1_b=['./Exp2/test_eval_metric_per_step_Study_Exp2_110ps.txt','./Exp2/test_eval_metric_per_step_Study_Exp2_110ps.txt']
filenames2_b=['./Exp2/test_eval_metric_per_step_Study_Exp2_160ps.txt','./Exp2/test_eval_metric_per_step_Study_Exp2_160ps.txt']

#------------------------------------------------------------------------------------------------------------#	

mean0_p,std0_p=mean_std(filenames0_p)
mean1_p,std1_p=mean_std(filenames1_p)
mean2_p,std2_p=mean_std(filenames2_p)


mean0_b,std0_b=mean_std(filenames0_b)
mean1_b,std1_b=mean_std(filenames1_b)
mean2_b,std2_b=mean_std(filenames2_b)


#------------------------------------------------------------------------------------------------------------#	

mean_p=[mean0_p,mean1_p,mean2_p]
std_p=[std0_p,std1_p,std2_p]

mean_b=[mean0_b,mean1_b,mean2_b]
std_b=[std0_b,std1_b,std2_b]

#------------------------------------------------------------------------------------------------------------#	

bar_width=.25
ind_p=[0,.75,1.5]
ind_b=[.25,1,1.75]

#------------------------------------------------------------------------------------------------------------#	

ax_proposed=plt.bar(ind_p,mean_p,width=bar_width,yerr=std_p,bottom=0,color='r',label='Exp1: Reaching')
ax_baseline=plt.bar(ind_b,mean_b,width=bar_width,yerr=std_b,bottom=0,color='g',label='Exp2: Pushing')
my_ticks=['60','110', '160']
x_ticks=[.125,.875,1.625]
plt.xticks(x_ticks,my_ticks)
plt.legend(loc='upper right')
#plt.title('Task completion measures for varying number of steps')
plt.xlabel('No: of steps per episode')
plt.ylim(0,1.5)
plt.xlim(-.5,2.25)
plt.ylabel('Task completion measure')
plt.savefig('completion_rate_vseps.png')
plt.show()
