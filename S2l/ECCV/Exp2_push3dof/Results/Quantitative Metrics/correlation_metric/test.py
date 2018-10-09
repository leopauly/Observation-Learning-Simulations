## @leopauly
## for calculating correaltio coefficient


import numpy as np

x=np.random.randint(0,50,1000)
y=x+np.random.normal(0,10,1000)

print(np.corrcoef(x,y))

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

plt.scatter(x,y)
plt.show()