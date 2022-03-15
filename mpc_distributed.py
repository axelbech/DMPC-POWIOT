#%%
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

from time import time 
from pickle import dumps, loads
from concurrent.futures import ProcessPoolExecutor

class MPC_single_home():
    def __init__(self, N, name, COP = 3.5, out_temp = 10, ref_temp = 23, P_max = 1.5):
        self.N = N
        self.name = name
        self.COP = COP
        self.out_temp = out_temp
        self.ref_temp = ref_temp
        self.P_max = P_max

        self.WallFunc, self.RoomFunc = self.get_dynamics_functions()
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([entry('Energy'), entry('Comfort')])
        p_model = struct_symMX([entry('rho_out'), entry('rho_in')])
        p_price = struct_symMX([entry('spot_price')])
        return struct_symMX([
                entry('Weights', struct=p_weights),
                entry('Model', struct=p_model),
                entry('Price', struct=p_price, repeat=self.N),
                entry('dual_variable', repeat=self.N-1)
            ])
        
        
    def get_dynamics_functions(self):
        Wall = MX.sym('Wall')
        Room = MX.sym('Room')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        WallPlus = Wall + rho_out * (OutTemp - Wall) + rho_in * (Room - Wall)
        WallFunc = Function('WallPlus', [rho_out, rho_in, Wall, Room, OutTemp], [WallPlus])

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        RoomPlus = Room + rho_in * (Wall - Room) + COP * Pow
        RoomFunc = Function('RoomPlus', [rho_in, Room, Wall, COP, Pow], [RoomPlus])

        return WallFunc, RoomFunc

    def get_decision_variables(self):
        MPC_states = struct_symMX([entry('Room'), entry('Wall')])
        MPC_inputs = struct_symMX([entry('P_hp')])

        w = struct_symMX([
        entry('State', struct=MPC_states, repeat=self.N),
        entry('Input', struct=MPC_inputs, repeat=self.N - 1)
        ])
        return w    
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N - 1):
            J += self.p['Weights', 'Energy'] * self.p['Price', k, 'spot_price'] * self.w['Input', k, 'P_hp']
            J += self.p['Weights', 'Comfort'] * (self.w['State', k, 'Room'] - self.ref_temp)**2
            
            J += self.p['dual_variable', k] * self.w['Input', k, 'P_hp'] # From dual decomposition, dual_var like a power price
            
        J += self.p['Weights', 'Comfort'] * (self.w['State', -1, 'Room'] - self.ref_temp)**2 # Accounting for the last time step state
        
        return J
    
    def get_constraint_functions(self):
        rho_out = self.p['Model', 'rho_out']
        rho_in = self.p['Model', 'rho_in']
        g = []
        lbg = []
        ubg = []
        
        for k in range(self.N - 1):
            Wall = self.w['State', k, 'Wall']
            Room = self.w['State', k, 'Room']
            Pow = self.w['Input', k, 'P_hp']
            
            WallPlus = self.WallFunc(rho_out, rho_in, Wall, Room, self.out_temp)
            RoomPlus = self.RoomFunc(rho_in, Room, Wall, self.COP, Pow)
            
            g.append(WallPlus - self.w['State', k+1, 'Wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(RoomPlus - self.w['State', k+1, 'Room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraints()
        mpc_problem = {'f': self.get_cost_funtion(), 'x': self.w, 'g': vertcat(*(g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def get_MPC_action(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        return solution['x']
    
def update_constraints(w0, lbw, ubw):
    lbw['State', 0, 'Wall'] = w0['State', 0, 'Wall']
    ubw['State', 0, 'Wall'] = w0['State', 0, 'Wall']
    
    lbw['State', 0, 'Room'] = w0['State', 0, 'Room']
    ubw['State', 0, 'Room'] = w0['State', 0, 'Room']
    
    lbw['Input', 0, 'P_hp'] = w0['Input', 0, 'P_hp']
    ubw['Input', 0, 'P_hp'] = w0['Input', 0, 'P_hp']
    
def update_initial_state(w0, x_0, N):
    w0['State', :N-1] = x_0['State', 1:]
    w0['State', -1] = x_0['State', -1]
    
    w0['Input', :N-2] = x_0['Input', 1:]
    w0['Input', -1] = x_0['Input', -1]
    
    
def prepare_MPC_action(w0, x_0, N, lbw, ubw):
    update_initial_state(w0, x_0, N)
    update_constraints(w0, lbw, ubw)
    

                
#%%

def price_func_exp(x):
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))

N = 288 # MPC horizon (how far it optimizes)
T = 10 # Running time (how many times do we solve opt. prob.)
spot_prices = np.fromfunction(price_func_exp, (N+T,)) # Spot prices for two days, 5 min intervals

home_list = ['axel', 'seb']
state_dict = dict.fromkeys(home_list, {})
ref_temp = {'axel': 21, 'seb': 24}
for home in state_dict:
    state_dict[home]["mpc"] = MPC(N, home, ref_temp = ref_temp[home])
    
    
state_0_axel = {'Wall': 9, 'Room': 10, 'P_hp': 0}
state_0_seb = {'Wall': 14, 'Room': 16, 'P_hp': 0}
# state_0_kang = {'Wall': 24, 'Room': 28, 'P_hp': 0}
state_0 = {'axel': state_0_axel, 'seb': state_0_seb}
n = 3 # Number of states

mpc = MPC(N, home, ref_temp=ref_temp)

#%%

x_0 = mpc.w(0)
for home in homes:
    x_0['State', :, home, 'Room'] = state_0[home]['Room']
    x_0['State', :, home, 'Wall'] = state_0[home]['Wall']
    x_0['Input', :, home, 'P_hp'] = state_0[home]['P_hp']

p_num = mpc.p(0)

p_num['Weights', 'Energy'] = 100
p_num['Weights', 'Comfort'] = 1
p_num['Weights', 'Peak'] = 20
p_num['Model', 'rho_out'] = 0.18
p_num['Model', 'rho_in'] = 0.37

# mpc.update_numerical_parameters(p_num)
#%%

traj_full = {}
# pickle.loads(pickle.dumps(mpc.w0))
# pickle.loads(pickle.dumps(mpc.p_num))
for home in homes:
    traj_full[home] = {'Room': [],'Wall': [],'P_hp': []}
    
traj_full['Peak'] = [0]
    
for home in homes:
    traj_full[home]['Room'].append(state_0[home]['Room'])
    traj_full[home]['Wall'].append(state_0[home]['Wall'])
    traj_full[home]['P_hp'].append(state_0[home]['P_hp'])
    traj_full['Peak'][0] += state_0[home]['P_hp']
    
if __name__ == '__main__':
    print("Starting calculations with horizon length =", N)
    for t in range(T-1):
        for i in range(N):
            p_num['Price', i, 'spot_price'] = spot_prices[t+i]
        mpc.prepare_MPC_action(x_0)
        # futures = []
        with ProcessPoolExecutor() as executor:
            future = executor.submit(mpc.get_MPC_action)
            x = future.result()
            
        # x = mpc.get_MPC_action()
        x = mpc.w(x) # Convert into struct with same structure as w
        for home in homes:
            traj_full[home]['Room'].append(x['State', 1, home, 'Room'])
            traj_full[home]['Wall'].append(x['State', 1, home, 'Wall'])
            traj_full[home]['P_hp'].append(x['Input', 1, home, 'P_hp'])
        traj_full['Peak'].append(x['Peak', 1])
        
        x_0 = x
        print("Iteration",t+1,"/",T,end="\r")
    
    print(traj_full['axel']['Room'])
