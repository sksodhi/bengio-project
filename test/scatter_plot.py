#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

#data = np.array([[0,1,1,1], [1,0,0,1], [1,1,1,0], [0,0,0,1]])
data = np.array([[0,1,2,1,1], [1,0,0,2,1], [1,1,1,0,1], [0,0,0,1,2]])

# get the indices where data is 1
x,y = np.argwhere(data == 1).T

print x 
print y 

plt.scatter(x,y)
plt.show()
#However, when you just want to visualize the 4x4 array you can use matshow

plt.matshow(data)
plt.show()
