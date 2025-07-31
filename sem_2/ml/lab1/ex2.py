#y=2x+3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x= np.linspace( -100,100,100)
y=(2*x)+3
plt.plot(x,y)
plt.show()
