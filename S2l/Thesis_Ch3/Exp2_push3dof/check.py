import numpy as np

y0 = np.loadtxt('action_store_0.txt', unpack=True)
y1 = np.loadtxt('action_store_1.txt', unpack=True)
y2 = np.loadtxt('action_store_2.txt', unpack=True)
y10 = np.loadtxt('action_store_10000.txt', unpack=True)

print(y2-y1)
