from casadi import *
from casadi.tools import *
import copy
import numpy as np
import matplotlib.pyplot as plt

homes = ['seb', 'axel']

N = 10

time = [x for x in range(N)]

l = []
figures = []

for idx, home in enumerate(homes):
    l.append([x**idx for x in range(N)])
    figures.append(plt.figure(home))
    plt.plot(time, l[idx])

plt.show()
    
