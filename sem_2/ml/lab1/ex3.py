#y=2x^2+3x+4

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x= np.linspace( -10,10,100)
y=(2*x*x)+(3*x)+4
plt.plot(x,y)
plt.show()