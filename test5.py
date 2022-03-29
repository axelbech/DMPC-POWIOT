from casadi import *
from casadi.tools import *
import numpy as np
from time import time
from pickle import dumps, loads

N = 5
avg_power = 1

x = struct_symMX([entry('P',repeat=N)])
lbx = x(0)
ubx = x(inf)

p = struct_symMX([entry('r',repeat=N)])

J = 0
power_sum = 0
for i in range(N):
    J += (x['P',i] - p['r',i])**2
    power_sum += x['P',i]
    
g = power_sum
lbg = 0
ubg =  avg_power * N

problem = {'f': J, 'x': x, 'g': g, 'p': p}
opts = {'ipopt.print_level':0, 'print_time':0}
solver = nlpsol('solver', 'ipopt', problem, opts)

x0 = x(avg_power)
refs = [avg_power + (i/N)  for i in range(N)]
p_num = p(refs)


solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_num)['x']

print(solution)