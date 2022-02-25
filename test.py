from casadi import *
from casadi.tools import *
import copy

N = 4

homes = ['seb', 'axel']


MPC_states = struct_symMX([1, 2])
MPC_inputs = struct_symMX([entry('P_hp')])

states_all = []
inputs_all = []
for home in homes:
    states_all.append(entry(home, struct=MPC_states))
    inputs_all.append(entry(home, struct=MPC_inputs))
states_all = struct_symMX(states_all)
inputs_all = struct_symMX(inputs_all)

w = struct_symMX([
entry('State', struct=states_all, repeat=N),
entry('Input', struct=inputs_all, repeat=N-1),
entry('Peak', repeat=N-1)
])