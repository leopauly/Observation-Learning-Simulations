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


mean0_e2,std0_e2=mean_std(.8442,.9515)
mean1_e2,std1_e2=mean_std(.6782,.8442)
mean2_e2,std2_e2=mean_std(.8235,.8650)
mean3_e2,std3_e2=mean_std(.8096,.8477)
mean4_e2,std4_e2=mean_std(.8892,.8615)
mean5_e2,std5_e2=mean_std(.7923,.6332)


mean0_e3,std0_e3=mean_std(1,1)
mean1_e3,std1_e3=mean_std(1,1)
mean2_e3,std2_e3=mean_std(1,1)
mean3_e3,std3_e3=mean_std(1,1)
mean4_e3,std4_e3=mean_std(1,1)
mean5_e3,std5_e3=mean_std(1,1)


#------------------------------------------------------------------------------------------------------------#	

mean_e2=[mean0_e2,mean1_e2,mean2_e2,mean3_e2,mean4_e2,mean5_e2]
std_e2=[std0_e2,std1_e2,std2_e2,std3_e2,std4_e2,std5_e2]

mean_e3=[mean0_e3,mean1_e3,mean2_e3,mean3_e3,mean4_e3,mean5_e3]
std_e3=[std0_e3,std1_e3,std2_e3,std3_e3,std4_e3,std5_e3]

#------------------------------------------------------------------------------------------------------------#	

bar_width=1
ind_e3=[0,4,8,12,16,20]
ind_e2=[1,5,9,13,17,21]

#------------------------------------------------------------------------------------------------------------#	

ax_e2=plt.bar(ind_e2,mean_e2,width=bar_width,yerr=std_e2,bottom=0,color='r',label='Pushing')
ax_e3=plt.bar(ind_e3,mean_e3,width=bar_width,yerr=std_e3,bottom=0,color='g',label='Hammering')
my_ticks=['V1','V2', 'Obj1','Obj2', 'BG','M']
x_ticks=[1,5,9,13,17,21]
plt.xticks(x_ticks,my_ticks)
plt.legend(loc='upper right')
plt.title('Tasks: Reaching and Hammering')
plt.xlabel('Experiments')
plt.ylim(-.5,1.5)
plt.ylabel('Task completion measure')
plt.savefig('Completionrate_barplot_real_e2_e3.png')
plt.show()
