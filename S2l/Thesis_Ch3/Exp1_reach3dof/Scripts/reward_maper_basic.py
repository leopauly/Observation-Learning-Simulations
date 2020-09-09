#### Program for plotting reawrd as intesity on the evn image
#### Author : leopauly

## Imports
import cv2
import numpy as np
import sys

run=sys.argv[1]
env_img=cv2.imread('env_img.png')
print(env_img.shape)

pos_x_=-(0)
pos_y_=-(0)

pos_x_converted= (pos_x_-(-2.2))/ (2.2-(-2.2))* (640-0)+0
pos_y_converted= (pos_y_-(-.62)) / (2.1-(-.62)) * (0-(-360)) + (-360) #pos_y_converted= (pos_y_-(-1)) / (2.5-(-1)) * (0-(-360)) + (-360)


cv2.circle(env_img,(int(pos_x_converted),int(-pos_y_converted)),4,(0,0,2800),-1)

        

cv2.imshow('env_imag',env_img)
cv2.imwrite('env_img_map.png',env_img)
cv2.waitKey()




