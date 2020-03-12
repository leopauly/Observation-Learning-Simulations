#### Code for converting videos to frames


import pylab
import cv2
import imageio
import scipy 
from skimage.transform import rotate

filename = './task1.1.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num= 0
for i in range(0,100000):
    image = vid.get_data(i)
    image_resize=cv2.resize(image, (112,112))
    #image_rot=rotate(image_resisze,0)
    cv2.imwrite("%d.png" % num,image_resize)
    num=num+1
    print(i)
    
   
