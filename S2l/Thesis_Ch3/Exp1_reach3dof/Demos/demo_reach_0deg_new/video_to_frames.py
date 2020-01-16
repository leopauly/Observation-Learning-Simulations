#### Code for converting videos to frames


import pylab
import cv2
import imageio
import scipy 
from skimage.transform import rotate

filename = './reach.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num= 0
for i in range(0,20000,1):
    image = vid.get_data(i)
    
    image_resize=cv2.resize(image, (112,112))    
    #rows,cols,_ = image.shape
    #scipy.misc.imwrite("%d.png" % i,image)
    #M = cv2.getRotationMatrix2D((cols,rows),90,1)
    #dst = cv2.warpAffine(image,M,(cols/2,rows/2))
    image_rot=rotate(image_resize,90)

    imageio.imwrite("./%03d.png" % num,image_rot)
    #cv2.imwrite("./%03d.png" % num,image)
    num=num+1
    print(i)
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(i), fontsize=20)
    #pylab.imshow(image)
    #pylab.show()
   
