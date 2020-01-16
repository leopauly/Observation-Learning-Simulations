## @leopauly
## For finding correlation between rewards per episode vs eval_metric per episode


## loading values
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def corr_coef(filename_x,filename_y,i):
    #-------------------------------------------------------------------------------------------------#

    y = np.loadtxt(filename_y, unpack=True)
    y_new=[y_ for y_ in y if y_!=0]
    print('y size:' ,np.array(y_new).shape)

    #-------------------------------------------------------------------------------------------------#

    x = np.loadtxt(filename_x, unpack=True)
    x_new=[x_ for x_ in x if x_!=0]

    x_norm=[]
    for i in range(len(x_new)):
        x_norm.append(1-(x_new[i]/x_new[0]))	
    print('x size:' ,np.array(x_new).shape)


    #-------------------------------------------------------------------------------------------------#

    ## correlation coefficient
    cor_coef=np.corrcoef(x_norm,y_new)
    print(np.corrcoef(x_norm,y_new))


    plt.scatter(x_norm,y_new,color='red')
    plt.hold(True)

    #-------------------------------------------------------------------------------------------------#
   
    return cor_coef



filenames_x_array=["eval_metric_per_epispde_run_2.txt","eval_metric_per_epispde_run_7.txt"]
filemames_y_array=["episode_reward_run_2.txt","episode_reward_run_7.txt"]
corr_per_run=[]
for i in range(len(filenames_x_array)):
    filename_x=filenames_x_array[i]
    filename_y=filemames_y_array[i]
    corr_per_run.append(corr_coef(filename_x,filename_y,i)[0][1])

corr_per_run=np.array(corr_per_run)
print('corr_per_run',corr_per_run)
mean_corr=np.mean(corr_per_run)
std_corr=np.std(corr_per_run)
print('Mean correlation cofficient:',mean_corr)
print('Std correlation cofficient:',std_corr)


plt.title('Visual reward vs Auxiliary')
plt.xlabel('Auxiliary reward')
plt.ylabel('Visual reward')
plt.savefig('Visual reward vs Auxiliary.png')
plt.show()
