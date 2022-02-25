#%%
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

import time

class MPC():
    def __init__(self, N, COP = 3.5, out_temp = 10, ref_temp = 23, P_max = 1.5):
        self.N = N
        self.COP = COP
        self.out_temp = out_temp
        self.ref_temp = ref_temp
        self.P_max = P_max
        
        self.WallFunc, self.RoomFunc = self.get_dynamics_functions()
        
        self.w = self.get_decision_variables()
        self.n_states = self.w['State', 0].numel()
        
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
        MPCstates = struct_symMX([entry('Room'), entry('Wall'), entry('Peak')])
        MPCinputs = struct_symMX([entry('P_hp')])

        w = struct_symMX([
        entry('State', struct=MPCstates, repeat=self.N),
        entry('Input', struct=MPCinputs, repeat=self.N - 1),
        ])
        return w
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N - 1):
            J += self.p['Weights', 'Energy'] * self.p['Price', k, 'spot_price'] * self.w['Input', k, 'P_hp']
            J += self.p['Weights', 'Comfort'] * (self.w['State', k, 'Room'] - self.ref_temp)**2
            
        J += self.p['Weights', 'Comfort'] * (self.w['State', k+1, 'Room'] - self.ref_temp)**2 # Accounting for the last time step state
        
        J += self.p['Weights', 'Peak'] * self.N * self.w['State', -1, 'Peak']
        return J
    
    def get_constraints(self):
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
            
            g.append(self.w['State', k+1, 'Peak'] - self.w['State', k, 'Peak'])
            lbg.append(0)
            ubg.append(inf)
            g.append(self.w['State', k, 'Peak'] - self.w['Input', k, 'P_hp'])
            lbg.append(0)
            ubg.append(inf)
            
        lbw = self.w(-inf)
        ubw = self.w(inf)
        lbw['Input', 1::, "P_hp"] = 0
        ubw['Input', 1::, "P_hp"] = self.P_max
            
        return g, lbg, ubg, lbw, ubw
    
    def get_solver(self):
        mpc_problem = {'f': self.J, 'x': self.w, 'g': vertcat(*(self.g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def update_initial_state(self, x_0):
        N = self.N
        n = self.n_states
        t_room = x_0[:n*N:n]
        t_wall = x_0[1:n*N:n]
        s_peak = x_0[2:n*N:n]
        P_hp = x_0[n*N:]
        
        for i in range(N-1):
            self.w0['State', i, 'Wall'] = t_wall[i+1]
            self.w0['State', i, 'Room'] = t_room[i+1]
            self.w0['State', i, 'Peak'] = s_peak[i+1]
        self.w0['State', -1, 'Wall'] = t_wall[-1]
        self.w0['State', -1, 'Room'] = t_room[-1]
        self.w0['State', -1, 'Peak'] = s_peak[-1]
        
        for i in range(N-2):
            self.w0['Input', i, 'P_hp'] = P_hp[i+1]
        self.w0['Input', -1, 'P_hp'] = P_hp[-1]
        
    def update_constraints(self):
        self.lbw['State', 0, 'Wall'] = self.w0['State', 0, 'Wall']
        self.ubw['State', 0, 'Wall'] = self.w0['State', 0, 'Wall']
        
        self.lbw['State', 0, 'Room'] = self.w0['State', 0, 'Room']
        self.ubw['State', 0, 'Room'] = self.w0['State', 0, 'Room']
        
        self.lbw['State', 0, 'Peak'] = self.w0['State', 0, 'Peak']
        self.ubw['State', 0, 'Peak'] = self.w0['State', 0, 'Peak']
        
        self.lbw['Input', 0, 'P_hp'] = self.w0['Input', 0, 'P_hp']
        self.ubw['Input', 0, 'P_hp'] = self.w0['Input', 0, 'P_hp']
        
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

N = 10 # MPC horizon (how far it optimizes)
T = 10 # Running time (how many times do we solve opt. prob.)
spot_prices = np.fromfunction(price_func_exp, (N+T,)) # Spot prices for two days, 5 min intervals

state_0 = {'Wall': 13, 'Room': 15, 'Peak': 0, 'P_hp': 0}
n = 3 # Number of states
x_0 = np.zeros(n*N + (N-1))
x_0[:n*N:n] = state_0['Room']
x_0[1:n*N:n] = state_0['Wall']
x_0[2:n*N:n] = state_0['Peak']
x_0[n*N:] = state_0['P_hp']

mpc = MPC(N)

p_num = mpc.p(0)

p_num['Weights', 'Energy'] = 100
p_num['Weights', 'Comfort'] = 1
p_num['Weights', 'Peak'] = 0.5 # 5
p_num['Model', 'rho_out'] = 0.18
p_num['Model', 'rho_in'] = 0.37
#%%
t_wall_full = [state_0['Wall']]
t_room_full = [state_0['Room']]
P_hp_full = [state_0['P_hp']]
s_peak_full = [state_0['Peak']]

print("Starting calculations with horizon length =", N)
for t in range(T-1):
    for i in range(N):
        p_num['Price', i, 'spot_price'] = spot_prices[t+i]
    x = mpc.get_MPC_action(p_num, x_0)
    t_room_full.append(x[3])
    t_wall_full.append(x[4])
    s_peak_full.append(x[5])
    P_hp_full.append(x[n*N + 1])
    
    x_0 = x
    print("Iteration",t+1,"/",T,end="\r")

time = [x for x in range(T)]

#%%

fig,ax=plt.subplots()
ax.plot(time, t_room_full, label="T_room")
ax.set_xlabel("5 minute intervals")
ax.set_ylabel("Temperature [°C]")
ax.plot()
ax.plot(time, t_wall_full, label="T_wall")
ax.legend()

axPwr = ax.twinx()
axPwr.plot(time, P_hp_full, label="P_hp", color="green")
axPwr.set_ylabel("Power [kW]")
axPwr.legend()
plt.show()
# %%
