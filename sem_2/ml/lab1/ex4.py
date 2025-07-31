#Gaussian PDF- mean=0, sigma=15, range(start=-100, stop=100, num=100):

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math as mt
x= np.linspace( -100,100,100)
sig=15
m=0
pi=3.14159
e=2.71828
y= (1/(sig*(mt.sqrt(2*pi))))*(e**(-0.5*((x-m)/sig)**2))
plt.plot(x,y)
plt.show()

