#%%
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

import time
import copy
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import os

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
        
        self.p = self.get_parameters_structure()
        
        self.WallFunc, self.RoomFunc = self.get_dynamics_functions()
        
        self.w = self.get_decision_variables()
        # self.n_states = self.w['State', 0, homes[0]].numel()
        # self.n_inputs = self.w['Input', 0, homes[0]].numel()
        self.n_homes = len(homes)
        
        # self.w0 = self.w(0)
        # self.J = self.get_cost_funtion()
        # self.g, self.lbg, self.ubg, self.lbw, self.ubw = self.get_constraints()
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([entry('Energy'), entry('Comfort'), entry('Peak')])
        p_model = struct_symMX([entry('rho_out'), entry('rho_in'), entry('COP')])
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
        for home in self.homes:
            for k in range(self.N - 1):
                Wall = self.w['State', k, home, 'Wall']
                Room = self.w['State', k, home, 'Room']
                Pow = self.w['Input', k, home, 'P_hp']
                
                WallPlus = self.WallFunc(rho_out, rho_in, Wall, Room, self.out_temp)
                RoomPlus = self.RoomFunc(rho_in, Room, Wall, self.p['Model','COP'], Pow)
                
                g.append(WallPlus - self.w['State', k+1, home, 'Wall'])
                lbg.append(0)
                ubg.append(0)
                g.append(RoomPlus - self.w['State', k+1, home, 'Room'])
                lbg.append(0)
                ubg.append(0)
        
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
                
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraints()
        mpc_problem = {'f': self.get_cost_funtion(), 'x': self.w, 'g': vertcat(*(g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def dummy_solver(self, w0, lbw, ubw, p_num):
        solver = self.get_solver()
        solution = solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        return solution['x']
        
    def get_MPC_action(self, w0, lbw, ubw, p_num):
        # then = time.time()

        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        print(os.getpid())
        
        # now = time.time()
        
        # print("Time elapsed = ", now-then, end="  ")

        return solution['x']
    
#%%
    
def update_constraints(w0, lbw, ubw, homes):
    for home in homes:
        lbw['State', 0, home, 'Wall'] = w0['State', 0, home, 'Wall']
        ubw['State', 0, home, 'Wall'] = w0['State', 0, home, 'Wall']
        
        lbw['State', 0, home, 'Room'] = w0['State', 0, home, 'Room']
        ubw['State', 0, home, 'Room'] = w0['State', 0, home, 'Room']
        
        lbw['Input', 0, home, 'P_hp'] = w0['Input', 0, home, 'P_hp']
        ubw['Input', 0, home, 'P_hp'] = w0['Input', 0, home, 'P_hp']
    
    lbw['Peak', 0] = w0['Peak', 0]
    ubw['Peak', 0] = w0['Peak', 0]
    
def update_initial_state(w0, x_0, N):
    w0['State', :N-1] = x_0['State', 1:]
    w0['State', -1] = x_0['State', -1]
    
    w0['Input', :N-2] = x_0['Input', 1:]
    w0['Input', -1] = x_0['Input', -1]
    
    w0['Peak', :N-2] = x_0['Peak', 1:]
    w0['Peak', -1] = x_0['Peak', -1]
    
def prepare_MPC_action(w0, x_0, N, homes, lbw, ubw):
    update_initial_state(w0, x_0, N)
    update_constraints(w0, lbw, ubw, homes)

def nlpsol_wrapper(sol, w0, lbw, ubw, lbg, ubg, p_num):
    res = sol(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p_num)
    print(os.getpid())
    return res['x']


                
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
w0 = mpc.w(0)
lbw = mpc.w(-inf)
ubw = mpc.w(inf)
    
for home in mpc.homes:
    lbw['Input', 1:, home, "P_hp"] = 0
    ubw['Input', 1:, home, "P_hp"] = mpc.P_max

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
p_num['Model', 'COP'] = 3.5

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
    # with Pool(processes=8) as pool:
    with ProcessPoolExecutor() as executor:
        for t in range(T-1):
            for i in range(N):
                p_num['Price', i, 'spot_price'] = spot_prices[t+i]
            prepare_MPC_action(w0, x_0, mpc.N, mpc.homes, lbw, ubw)
            futures = []
            R = 2
            # soll = [mpc.solver for _ in range(R)]
            w0l = [w0.master for _ in range(R)]
            lbwl = [lbw.master for _ in range(R)]
            ubwl = [ubw.master for _ in range(R)]
            # lbgl = [mpc.lbg for _ in range(R)]
            # ubgl = [mpc.ubg for _ in range(R)]
            p_numl = [p_num.master for _ in range(R)]
            # then = time.time()
            # # m = pool.starmap(mpc.get_MPC_action, zip(w0l, lbwl, ubwl, p_numl))
            m = executor.map(mpc.get_MPC_action, w0l, lbwl, ubwl, p_numl, chunksize=8)
            # now = time.time()
            # print("Time elapsed = ", now-then, end="  ")
            # for i in range(R):
            #     futures.append(executor.submit(mpc.get_MPC_action, w0.master, lbw.master, ubw.master, p_num.master))
                
            # x = list(m)[0]
            # print(x)
            for i in range(R):
                x = mpc.get_MPC_action(w0.master, lbw.master, ubw.master, p_num.master)
            x = mpc.w(x) # Convert into struct with same structure as w
            for home in homes:
                traj_full[home]['Room'].append(x['State', 1, home, 'Room'])
                traj_full[home]['Wall'].append(x['State', 1, home, 'Wall'])
                traj_full[home]['P_hp'].append(x['Input', 1, home, 'P_hp'])
            traj_full['Peak'].append(x['Peak', 1])
            
            x_0 = x
            print("Iteration",t+1,"/",T,end="\r")
            
    print(traj_full['axel']['Room'])
