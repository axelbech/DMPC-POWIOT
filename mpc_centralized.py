#%%
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

import time
import copy

class MPC():
    def __init__(self, N, homes, COP = 3.5, out_temp = 10, ref_temp = None, P_max = 1.5):
        self.N = N
        self.homes = homes
        self.COP = COP
        self.out_temp = out_temp
        if not ref_temp:
            self.ref_temp = dict((home,23) for home in homes)
        else:
            self.ref_temp = ref_temp
        self.ref_temp = ref_temp
        self.P_max = P_max
        
        self.WallFunc, self.RoomFunc = self.get_dynamics_functions()
        
        self.w = self.get_decision_variables()
        # self.n_states = self.w['State', 0, homes[0]].numel()
        # self.n_inputs = self.w['Input', 0, homes[0]].numel()
        self.n_homes = len(homes)
        
        self.w0 = self.w(0)
        self.p = self.get_parameters_structure()
        self.J = self.get_cost_funtion()
        self.g, self.lbg, self.ubg, self.lbw, self.ubw = self.get_constraints()
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([entry('Energy'), entry('Comfort'), entry('Peak')])
        p_model = struct_symMX([entry('rho_out'), entry('rho_in')])
        p_price = struct_symMX([entry('spot_price')])
        return struct_symMX([
                entry('Weights', struct=p_weights),
                entry('Model', struct=p_model),
                entry('Price', struct=p_price, repeat=self.N)
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
        
        states_all = []
        inputs_all = []
        for home in self.homes:
            states_all.append(entry(home, struct=MPC_states))
            inputs_all.append(entry(home, struct=MPC_inputs))
        states_all = struct_symMX(states_all)
        inputs_all = struct_symMX(inputs_all)

        w = struct_symMX([
        entry('State', struct=states_all, repeat=self.N),
        entry('Input', struct=inputs_all, repeat=self.N-1),
        entry('Peak', repeat=self.N-1)
        ])
        return w
    
    def get_cost_funtion(self):
        J = 0
        for home in self.homes:
            for k in range(self.N - 1):
                J += self.p['Weights', 'Energy'] * self.p['Price', k, 'spot_price'] * self.w['Input', k, home, 'P_hp']
                J += self.p['Weights', 'Comfort'] * (self.w['State', k, home, 'Room'] - self.ref_temp[home])**2
                
            J += self.p['Weights', 'Comfort'] * (self.w['State', k+1, home, 'Room'] - self.ref_temp[home])**2 # Accounting for the last time step state
        
        J += self.p['Weights', 'Peak'] * self.N * self.n_homes * self.w['Peak', -1]
        return J
    
    def get_constraints(self):
        rho_out = self.p['Model', 'rho_out']
        rho_in = self.p['Model', 'rho_in']
        g = []
        lbg = []
        ubg = []
        lbw = self.w(-inf)
        ubw = self.w(inf)
        for home in self.homes:
            for k in range(self.N - 1):
                Wall = self.w['State', k, home, 'Wall']
                Room = self.w['State', k, home, 'Room']
                Pow = self.w['Input', k, home, 'P_hp']
                
                WallPlus = self.WallFunc(rho_out, rho_in, Wall, Room, self.out_temp)
                RoomPlus = self.RoomFunc(rho_in, Room, Wall, self.COP, Pow)
                
                g.append(WallPlus - self.w['State', k+1, home, 'Wall'])
                lbg.append(0)
                ubg.append(0)
                g.append(RoomPlus - self.w['State', k+1, home, 'Room'])
                lbg.append(0)
                ubg.append(0)
            
            lbw['Input', 1:, home, "P_hp"] = 0
            ubw['Input', 1:, home, "P_hp"] = self.P_max
        
        for k in range(self.N - 2):
            g.append(self.w['Peak', k+1] - self.w['Peak', k])
            lbg.append(0)
            ubg.append(inf)
        for k in range(self.N - 1):
            power_sum = 0
            for home in self.homes:
                power_sum += self.w['Input', k, home, 'P_hp']
            g.append(self.w['Peak', k] - power_sum) # peak state must be greater than sum of power at k
            lbg.append(0)
            ubg.append(inf)
                
        return g, lbg, ubg, lbw, ubw
    
    def get_solver(self):
        mpc_problem = {'f': self.J, 'x': self.w, 'g': vertcat(*(self.g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def update_initial_state(self, x_0):
        self.w0['State', :self.N-1] = x_0['State', 1:]
        self.w0['State', -1] = x_0['State', -1]
        
        self.w0['Input', :self.N-2] = x_0['Input', 1:]
        self.w0['Input', -1] = x_0['Input', -1]
        
        self.w0['Peak', :self.N-2] = x_0['Peak', 1:]
        self.w0['Peak', -1] = x_0['Peak', -1]
        
    def update_constraints(self):
        for home in self.homes:
            self.lbw['State', 0, home, 'Wall'] = self.w0['State', 0, home, 'Wall']
            self.ubw['State', 0, home, 'Wall'] = self.w0['State', 0, home, 'Wall']
            
            self.lbw['State', 0, home, 'Room'] = self.w0['State', 0, home, 'Room']
            self.ubw['State', 0, home, 'Room'] = self.w0['State', 0, home, 'Room']
            
            self.lbw['Input', 0, home, 'P_hp'] = self.w0['Input', 0, home, 'P_hp']
            self.ubw['Input', 0, home, 'P_hp'] = self.w0['Input', 0, home, 'P_hp']
        
        self.lbw['Peak', 0] = self.w0['Peak', 0]
        self.ubw['Peak', 0] = self.w0['Peak', 0]
        
    def get_MPC_action(self, p_num, x_0):
        then = time.time()
        
        self.update_initial_state(x_0)
        self.update_constraints()

        solution = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        
        now = time.time()
        
        print("Time elapsed = ", now-then, end="  ")

        return solution['x']
    

                
#%%

def price_func_exp(x):
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))

N = 288 # MPC horizon (how far it optimizes)
T = 20 # Running time (how many times do we solve opt. prob.)
spot_prices = np.fromfunction(price_func_exp, (N+T,)) # Spot prices for two days, 5 min intervals
homes = ['axel', 'seb']
ref_temp = {'axel': 21, 'seb': 24}

state_0_axel = {'Wall': 9, 'Room': 10, 'P_hp': 0}
state_0_seb = {'Wall': 14, 'Room': 16, 'P_hp': 0}
# state_0_kang = {'Wall': 24, 'Room': 28, 'P_hp': 0}
state_0 = {'axel': state_0_axel, 'seb': state_0_seb}
n = 3 # Number of states

mpc = MPC(N, homes, ref_temp=ref_temp)

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
#%%

traj_full = {}
for home in homes:
    traj_full[home] = {'Room': [],'Wall': [],'P_hp': []}
    
traj_full['Peak'] = [0]
    
for home in homes:
    traj_full[home]['Room'].append(state_0[home]['Room'])
    traj_full[home]['Wall'].append(state_0[home]['Wall'])
    traj_full[home]['P_hp'].append(state_0[home]['P_hp'])
    traj_full['Peak'][0] += state_0[home]['P_hp']
    

print("Starting calculations with horizon length =", N)
for t in range(T-1):
    for i in range(N):
        p_num['Price', i, 'spot_price'] = spot_prices[t+i]
    x = mpc.get_MPC_action(p_num, x_0)
    x = mpc.w(x) # Convert into struct with same structure as w
    for home in homes:
        traj_full[home]['Room'].append(x['State', 1, home, 'Room'])
        traj_full[home]['Wall'].append(x['State', 1, home, 'Wall'])
        traj_full[home]['P_hp'].append(x['Input', 1, home, 'P_hp'])
    traj_full['Peak'].append(x['Peak', 1])
    
    x_0 = x
    print("Iteration",t+1,"/",T,end="\r")

time = [x for x in range(T)]

#%% 

# for idx, home in enumerate(homes):
#     # plt.figure(home)
#     fig,ax=plt.subplots(num=home)
#     ax.plot(time, traj_full[home]['Room'], label="T_room")
#     ax.set_xlabel("5 minute intervals")
#     ax.set_ylabel("Temperature [°C]")
#     ax.plot()
#     ax.plot(time, traj_full[home]['Wall'], label="T_wall")
#     ax.legend()

#     axPwr = ax.twinx()
#     axPwr.plot(time, traj_full[home]['P_hp'], label="P_hp", color="green")
#     axPwr.set_ylabel("Power [kW]")
#     axPwr.legend()
    
# plt.show()

fig,ax=plt.subplots()
for idx, home in enumerate(homes):
    ax.plot(time, traj_full[home]['P_hp'], label=home)
    ax.set_ylabel("Power [kW]")
    ax.legend()
ax.set_title('Power consumption, centralized approach')
plt.show()
