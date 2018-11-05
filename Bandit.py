import sys
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(-(10*np.sin(x)-5)**2/20)*np.sin(x)/(abs(x)+1)


#x = float(sys.argv[1])
#print(f(x))
x = np.linspace(-10,10)
plt.plot(x,f(x))
plt.show()
