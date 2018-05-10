import imageio
from os import listdir
from os.path import isfile, join

mypath='./images/ex2ep180/'
filenames_ = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filenames=sorted(filenames_)

print(filenames)

with imageio.get_writer('ex2ep180.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(join(mypath,filename))
        writer.append_data(image)
