#### For plotting task completion rates in bar plot for simulation experiments


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



def mean_std(rate1,rate2):
	
	mean=np.mean([rate1,rate2])
	std=np.std([rate1,rate2])
	return mean,std


#------------------------------------------------------------------------------------------------------------#	


mean0_e2,std0_e2=mean_std(1,1)
mean1_e2,std1_e2=mean_std(1,1)
mean2_e2,std2_e2=mean_std(1,1)

mean0_e3,std0_e3=mean_std(.75,1)
mean1_e3,std1_e3=mean_std(.875,0.75)
mean2_e3,std2_e3=mean_std(0.50,1)


#------------------------------------------------------------------------------------------------------------#	

mean_e2=[mean0_e2,mean1_e2,mean2_e2]
std_e2=[std0_e2,std1_e2,std2_e2]

mean_e3=[mean0_e3,mean1_e3,mean2_e3]
std_e3=[std0_e3,std1_e3,std2_e3]

#------------------------------------------------------------------------------------------------------------#	

bar_width=.5
ind_e3=[0,4,8]
ind_e2=[.5,4.5,8.5]

#------------------------------------------------------------------------------------------------------------#	

ax_e2=plt.bar(ind_e2,mean_e2,width=bar_width,yerr=std_e2,bottom=0,color='r',label='Striking')
ax_e3=plt.bar(ind_e3,mean_e3,width=bar_width,yerr=std_e3,bottom=0,color='g',label='Sweeping')
my_ticks=['V1','V2', 'BG']
x_ticks=[.25,4.52,8.25]
plt.xticks(x_ticks,my_ticks)
plt.legend(loc='upper right')
plt.title('Tasks: Sweeping and Striking')
plt.xlabel('Experiments')
plt.ylim(-.5,1.5)
plt.ylabel('Task completion measure')
plt.savefig('Completionrate_barplot_real_e4_e5.png')
plt.show()
