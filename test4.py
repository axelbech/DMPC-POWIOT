from casadi import *
from casadi.tools import *

N = 20
M = 5
p_max = 1.5

w = struct_symMX([entry('peak', repeat=N)])

p = struct_symMX([entry('weight'), entry('dual_variable', repeat=N)])

J = 0
J += M * N * p['weight'] * w['peak', -1]
for k in range(N):
    J -= p['dual_variable', k] * w['peak', k]
    
g = []
lbg = []
ubg = []
lbw = []
ubw = []
for k in range(N - 1):
    g.append(w['peak', k+1] - w['peak', k])
    lbg.append(0)
    ubg.append(inf)
for k in range(N):
    lbw.append(0)
    ubw.append(M * p_max)

mpc_problem = {'f': J, 'x': w, 'g': vertcat(*(g)), 'p': p}
opts = {'ipopt.print_level':0, 'print_time':0}
solver = nlpsol('solver', 'ipopt', mpc_problem, opts)

w0 = w(1)
p_num = p(0)
p_num['weight'] = 1
p_num['dual_variable', :] = 1
# p_num['dual_variable', 4] = 5

solution = solver(x0=w0,lbx=lbw,ubx=ubw,lbg=lbg,ubg=ubg,p=p_num) 
x = solution['x']

print(x)