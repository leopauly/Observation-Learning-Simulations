#### Code for converting videos to frames


import pylab
import cv2
import imageio
import scipy 
from skimage.transform import rotate

filename = './demo_push_1.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num= 0
for i in range(100,1000):
    image = vid.get_data(i)
    
    #rows,cols,_ = image.shape
    #scipy.misc.imwrite("%d.png" % i,image)
    #imageio.imwrite("%d.png" % i,image)
    #M = cv2.getRotationMatrix2D((cols,rows),90,1)
    #dst = cv2.warpAffine(image,M,(cols/2,rows/2))
    
    image_resize=cv2.resize(image, (112,112))
    #image_rot=rotate(image_resisze,0)
    cv2.imwrite("%d.png" % num,image_resize)
    num=num+1
    print(i)
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(i), fontsize=20)
    #pylab.imshow(image)
    #pylab.show()
   
