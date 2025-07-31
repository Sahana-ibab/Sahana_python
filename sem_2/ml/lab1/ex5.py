import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
x= np.linspace( -100,100,100)
y= x**2
d=2*x
plt.plot(x,y,d)
plt.show()
