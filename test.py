import time
from casadi import *
from casadi.tools import *

import concurrent.futures
from multiprocessing import Pool
import pickle
import os

R = 64
la = [x for x in range(R)]
lb = [2*x for x in range(R)]

def func(a, b):
    print(os.getpid())
    return a + b

if __name__ == '__main__':
    pool = Pool(processes=8)
    m = pool.starmap(func, zip(la, lb))
    print(m) 