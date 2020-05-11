#### Program for plotting reawrd as intesity on the evn image
#### Author : leopauly

## Imports
import numpy as np
import sys
import cv2
from matplotlib import cm
import matplotlib
import os

run=sys.argv[1]
env_img=cv2.imread('env_img.png')
intensity_multiplier=10
top_best=10


##----------------------------------------------------------------------------------------------------##

rewards_all=[]
pos_x_all=[]
pos_y_all=[]

for i in range(int(run)):
        
        rewards_ = np.loadtxt('episode_reward_run_'+str(i)+'.txt', unpack=True)
        rewards=[r_ for r_ in rewards_ if r_!=0]
        rewards_all=rewards_all+rewards

        pos_x_temp=np.array(np.loadtxt('episode_man_x_pos_run_'+str(i)+'.txt', unpack=True))
        pos_x=[pos for pos in pos_x_temp if pos!=0]
        pos_x_all=pos_x_all+pos_x

        pos_y_temp=np.array(np.loadtxt('episode_man_y_pos_run_'+str(i)+'.txt', unpack=True))
        pos_y=[pos for pos in pos_y_temp if pos!=0]
        pos_y_all=pos_y_all+pos_y


pos_x_temp=0
pos_y_temp=0
pos_x=0
pos_x=0
rewards_=0
rewards=0

##----------------------------------------------------------------------------------------------------##
rewards_all=np.array(rewards_all)
pos_x_all=np.array(pos_x_all)
pos_y_all=np.array(pos_y_all)
print('Shapes:',rewards_all.shape,pos_x_all.shape,pos_x_all.shape)

rewards_norm_all=[]
def normalise_data(data):
        return (data-np.min(data))/(np.max(data)-np.min(data))
rewards_norm_all=np.array(normalise_data(rewards_all))



##----------------------------------------------------------------------------------------------------##




assert(pos_x_all.shape==pos_y_all.shape==rewards_norm_all.shape),'Donot have equal number of x and y coordinates'
for k in range(len(pos_x_all)):

            pos_x_point=-(pos_x_all[k])
            pos_x_converted=(pos_x_point-(-2.2))/ (2.2-(-2.2))* (640-0)+0
            pos_y_point=-(pos_y_all[k])
            pos_y_converted=(pos_y_point-(-.62)) / (2.1-(-.62)) * (0-(-360)) + (-360) #pos_y_converted= (pos_y_-(-1)) / (2.5-(-1)) * (0-(-360)) + (-360
            color_value=cm.jet(rewards_norm_all[k]) 
            color_value_rgb=(255*color_value[2],255*color_value[1],255*color_value[0]) # (0,0,255*rewards_norm_all[k]) 
            print(color_value)
            cv2.circle(env_img,(int(pos_x_converted),int(-pos_y_converted)),4,color_value_rgb,-1)
print()

##----------------------------------------------------------------------------------------------------##       

rewards_norm_all_sorted_idx=np.argsort(rewards_all)[::-1]
#print('Sorted_rewards:',rewards_all[rewards_norm_all_sorted_idx])


for l in rewards_norm_all_sorted_idx[0:top_best]:
        print('Top rewards:',rewards_all[l],rewards_norm_all[l])
        pos_x_point=-(pos_x_all[l])
        pos_x_converted=(pos_x_point-(-2.2))/ (2.2-(-2.2))* (640-0)+0
        pos_y_point=-(pos_y_all[l])
        pos_y_converted=(pos_y_point-(-.62)) / (2.1-(-.62)) * (0-(-360)) + (-360)
        cv2.circle(env_img,(int(pos_x_converted),int(-pos_y_converted)),8,(0,0,255),1)

rewards_norm_all_sorted_idx=np.argsort(rewards_all)
#print('Sorted_rewards:',rewards_all[rewards_norm_all_sorted_idx])


for l in rewards_norm_all_sorted_idx[0:top_best]:
        print('Lowest rewards:',rewards_all[l],rewards_norm_all[l])
        pos_x_point=-(pos_x_all[l])
        pos_x_converted=(pos_x_point-(-2.2))/ (2.2-(-2.2))* (640-0)+0
        pos_y_point=-(pos_y_all[l])
        pos_y_converted=(pos_y_point-(-.62)) / (2.1-(-.62)) * (0-(-360)) + (-360)
        cv2.circle(env_img,(int(pos_x_converted),int(-pos_y_converted)),8,(0,0,0),1)

##----------------------------------------------------------------------------------------------------##
cv2.imshow('env_img_map',env_img)
cv2.imwrite('env_img_map_all_best_cmap.png',env_img)
cv2.waitKey()




