from casadi import *
from casadi.tools import *
import numpy as np
from time import time
from pickle import dumps, loads
import matplotlib.pyplot as plt




x = struct_symMX([entry('P')])
lbx = x(0)
ubx = x(inf)

p = struct_symMX([entry('r'), entry('dual_variable')])

J = (x['P']-p['r'])**2 + p['dual_variable'] * x['P']

problem = {'f': J, 'x': x, 'p': p}
opts = {'ipopt.print_level':0, 'print_time':0}
solver = nlpsol('solver', 'ipopt', problem, opts)
    
    
home_list = ['axel', 'seb', 'simen', 'ida', 'kang']
N = len(home_list)
avg_power = 1
L = avg_power * N

state_dict = {}
refs = [avg_power + (i/N)  for i in range(N)]
r = {home:reference for home, reference in zip(home_list, refs)}

xs = [x(avg_power) for _ in home_list]
ps = [p(r) for r in refs]
state_dict = {home:{'x':x, 'p':p} for home, x, p in zip(home_list, xs, ps)}

def update_dual_variable(state_dict, dual_variable, alpha):
    power_sum = 0
    for home_dict in state_dict.values():
        power_sum += home_dict['x']['P']
    
    dual_variable += alpha * (power_sum - L)
    
    for home_dict in state_dict.values():
        home_dict['p']['dual_variable'] = dual_variable
    
    return dual_variable

def update_state(state_dict, result_list):
    for res_home, home_dict in zip(result_list, state_dict.values()):
        home_dict['x'] = res_home
        
    
        
p_num = p(0)
p_num['r'] = 1.5
p_num['dual_variable'] = 1

dual_variable = 1
dv_traj = [dual_variable]
k = 1
while True:
    dual_variable_last = dual_variable

    result_list = []
    for hd in state_dict.values():
        solution = solver(x0=hd['x'],lbx=lbx,ubx=ubx,p=hd['p'])
        result_list.append(x(solution['x']))
        
    update_state(state_dict, result_list)

    alpha = 0.5 / sqrt(k)
    dual_variable = update_dual_variable(state_dict, dual_variable, alpha)
    dv_traj.append(dual_variable)

    dv_diff = np.abs(dual_variable - dual_variable_last)
    print(f'Dual variable change = {dv_diff}')
    if dv_diff < 0.01:
        break
    k+=1

figdv, axdv = plt.subplots()
axdv.plot(dv_traj)
axdv.set_title('Dual variable')