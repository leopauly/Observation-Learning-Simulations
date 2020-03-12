#### Code for converting videos to frames


import pylab
import cv2
import imageio
import scipy 
from skimage.transform import rotate

filename = './vid.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num= 0
for i in range(0,20000,1):
    image = vid.get_data(i)
    
    image_resize=cv2.resize(image, (112,112))    
    image_rot=rotate(image_resize,0)

    imageio.imwrite("./%03d.png" % num,image_rot)
    num=num+1
 
