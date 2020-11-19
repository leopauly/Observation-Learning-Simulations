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

mean0_e4,std0_e4=mean_std(1,1)
mean1_e4,std1_e4=mean_std(1,1)
mean2_e4,std2_e4=mean_std(1,1)

mean0_e5,std0_e5=mean_std(.75,1)
mean1_e5,std1_e5=mean_std(.875,0.75)
mean2_e5,std2_e5=mean_std(0.50,1)


#------------------------------------------------------------------------------------------------------------#	

mean_e2=[mean0_e2,mean1_e2,mean2_e2,mean3_e2,mean4_e2,mean5_e2]
std_e2=[std0_e2,std1_e2,std2_e2,std3_e2,std4_e2,std5_e2]

mean_e3=[mean0_e3,mean1_e3,mean2_e3,mean3_e3,mean4_e3,mean5_e3]
std_e3=[std0_e3,std1_e3,std2_e3,std3_e3,std4_e3,std5_e3]


mean_e4=[mean0_e4,mean1_e4,mean2_e4]
std_e4=[std0_e4,std1_e4,std2_e4]

mean_e5=[mean0_e5,mean1_e5,mean2_e5]
std_e5=[std0_e5,std1_e5,std2_e5]


#------------------------------------------------------------------------------------------------------------#	

bar_width=1
ind_e2=[0,5,10,13,16,21]
ind_e3=[1,6,11,14,17,22]
ind_e5=[2,7,18]
ind_e4=[3,8,19]


#------------------------------------------------------------------------------------------------------------#	

ax_e2=plt.bar(ind_e2,mean_e2,width=bar_width,yerr=std_e2,bottom=0,color='r',label='Pushing')
ax_e3=plt.bar(ind_e3,mean_e3,width=bar_width,yerr=std_e3,bottom=0,color='g',label='Hammering')
ax_e5=plt.bar(ind_e5,mean_e5,width=bar_width,yerr=std_e5,bottom=0,color='goldenrod',label='Sweeping')
ax_e4=plt.bar(ind_e4,mean_e4,width=bar_width,yerr=std_e4,bottom=0,color='cornflowerblue',label='Striking')
my_ticks=['I','V', 'Obj','Obj', 'BG','M']
x_ticks=[1.5,6.5,10.5,13.5,17.5,21.5]
plt.xticks(x_ticks,my_ticks)
plt.legend(loc='upper right')
plt.title('')
plt.xlabel('Domain shifts')
plt.ylim(-0,1.5)
plt.ylabel('Task completion measure')
plt.savefig('Completionrate_barplot_real_e2_e3_e4_e5.png')
plt.show()
