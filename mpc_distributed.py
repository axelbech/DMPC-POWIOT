#%%
from importlib.abc import MetaPathFinder
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

from time import time 
from pickle import dumps, loads
from concurrent.futures import ProcessPoolExecutor

class MPC_single_home():
    def __init__(self, N, P_max = 1.5):
        self.N = N
        # self.name = name
        # self.out_temp = out_temp
        # self.ref_temp = ref_temp
        self.P_max = P_max

        self.wall_func, self.room_func = self.get_dynamics_functions()
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([entry('energy'), entry('comfort')])
        p_model = struct_symMX([entry('rho_out'), entry('rho_in'), entry('COP')])
        return struct_symMX([
                entry('weights', struct=p_weights),
                entry('model', struct=p_model),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('dual_variable', repeat=self.N-1)
            ])
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall')
        room = MX.sym('room')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function('wall_plus', [rho_out, rho_in, wall, room, OutTemp], [wall_plus])

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function('room_plus', [rho_in, room, wall, COP, Pow], [room_plus])

        return wall_func, room_func

    def get_decision_variables(self):
        MPC_states = struct_symMX([entry('room'), entry('wall')])
        MPC_inputs = struct_symMX([entry('P_hp')])

        w = struct_symMX([
        entry('state', struct=MPC_states, repeat=self.N),
        entry('input', struct=MPC_inputs, repeat=self.N - 1)
        ])
        return w    
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N): 
            J += self.p['weights', 'comfort'] * (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['weights', 'energy'] * self.p['spot_price', k] * self.w['input', k, 'P_hp']
            J += self.p['dual_variable', k] * self.w['input', k, 'P_hp'] # From dual decomposition, dual_var like a power price
        
        return J
    
    def get_constraint_functions(self):
        rho_out = self.p['model', 'rho_out']
        rho_in = self.p['model', 'rho_in']
        g = []
        lbg = []
        ubg = []
        
        for k in range(self.N - 1):
            wall = self.w['state', k, 'wall']
            room = self.w['state', k, 'room']
            Pow = self.w['input', k, 'P_hp']
            out_temp = self.p['outdoor_temperature', k]
            
            wall_plus = self.wall_func(rho_out, rho_in, wall, room, out_temp)
            room_plus = self.room_func(rho_in, room, wall, self.p['model', 'COP'], Pow)
            
            g.append(wall_plus - self.w['state', k+1, 'wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {'f': self.get_cost_funtion(), 'x': self.w, 'g': vertcat(*(g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def get_MPC_action(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        return solution['x']
    
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        lbw['state', 0, 'wall'] = w0['state', 0, 'wall']
        ubw['state', 0, 'wall'] = w0['state', 0, 'wall']
        
        lbw['state', 0, 'room'] = w0['state', 0, 'room']
        ubw['state', 0, 'room'] = w0['state', 0, 'room']
        
        lbw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        ubw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        
    @staticmethod
    def update_initial_state(w0, x_0, N):
        w0['state', :N-1] = x_0['state', 1:]
        w0['state', -1] = x_0['state', -1]
        
        w0['input', :N-2] = x_0['input', 1:]
        w0['input', -1] = x_0['input', -1]
        
    @staticmethod   
    def prepare_MPC_action(w0, x_0, N, lbw, ubw):
        MPC_single_home.update_initial_state(w0, x_0, N)
        MPC_single_home.update_constraints(w0, lbw, ubw)
    
class MPC_peak_state():
    def __init__(self, n_steps, n_homes, P_max=1.5):
        self.n_steps = n_steps
        self.n_homes = n_homes
        self.P_max = P_max
        
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.solver = self.get_solver()
        
    def get_decision_variables(self):
        return struct_symMX([entry('peak', repeat=self.n_steps)])
    
    def get_parameters_structure(self):
        return struct_symMX([entry('weight'), entry('dual_variable', repeat=self.n_steps)])

    def get_cost_function(self):
        J = 0
        J += self.n_homes * self.n_steps * self.p['weight'] * self.w['peak', -1]
        for k in range(self.n_steps):
            J -= self.p['dual_variable', k] * self.w['peak', k]
        return J
            
    def get_constraint_functions(self):
        g = []
        lbg = []
        ubg = []
        for k in range(self.n_steps - 1):
            g.append(self.w['peak', k+1] - self.w['peak', k])
            lbg.append(0)
            ubg.append(inf)
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {'f': self.get_cost_function(), 'x': self.w, 'g': vertcat(*(g)), 'p': self.p}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def solve_peak_problem(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                                lbg=self.lbg, ubg=self.ubg, p=p_num)
        return solution['x']
    
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        lbw['peak', 0] = w0['peak', 0]
        ubw['peak', 0] = w0['peak', 0] 
        
    @staticmethod
    def update_initial_state(w0, x_0, n_steps):
        w0['peak', :n_steps-1] = x_0['peak', 1:]
        w0['peak', -1] = x_0['peak', -1]
        
    @staticmethod   
    def prepare_action(w0, x_0, n_steps, lbw, ubw):
        MPC_peak_state.update_initial_state(w0, x_0, n_steps)
        MPC_peak_state.update_constraints(w0, lbw, ubw)

class DMPC():
    def __init__(self, N, T, home_list, state_0, spot_prices, outdoor_temperature):
        self.N = N
        self.T = T
        self.home_list = home_list
        self.state_dict = self.build_state_dict(N, T, home_list, state_0, spot_prices, outdoor_temperature)
        self.executor = ProcessPoolExecutor()
        
    @staticmethod
    def build_state_dict(N, T, home_list, state_0, spot_prices, outdoor_temperature):
        mpc_single_home = MPC_single_home(N)
        state_dict = {'mpc_single_home': mpc_single_home, 'homes': dict.fromkeys(home_list, {}),  'peak': {}}
        for home, home_dict in zip(home_list, state_dict['homes'].values()):
            # mpc = MPC_single_home(N, home)
            
            p_num = mpc_single_home.p(0)
            p_num['outdoor_temperature', :] = list(outdoor_temperature)
            p_num['reference_temperature', :] = list(ref_temp[home])
            p_num['spot_price', :] = list(spot_prices[:N-1])
            p_num['weights', 'energy'] = 100
            p_num['weights', 'comfort'] = 1
            p_num['model', 'rho_out'] = 0.18
            p_num['model', 'rho_in'] = 0.37
            
            w0 = mpc_single_home.w(0)
            lbw = mpc_single_home.w(-inf)
            ubw = mpc_single_home.w(inf)
            lbw['input', :, 'P_hp'] = 0
            ubw['input', :, 'P_hp'] = mpc_single_home.P_max
            
            x = mpc_single_home.w(0)
            x['state', :, 'room'] = state_0[home]['room']
            x['state', :, 'wall'] = state_0[home]['wall']
            x['input', :, 'P_hp'] = state_0[home]['P_hp']
            
            traj_full = {'wall': np.zeros(T), 'room': np.zeros(T), 'P_hp': np.zeros(T)}
            traj_full['room'][0] = state_0[home]['room']
            traj_full['wall'][0] = state_0[home]['wall']
            traj_full['P_hp'][0] = state_0[home]['P_hp']
            
            home_dict['x'] = x
            home_dict['p_num'] = p_num
            home_dict['w0'] = w0
            home_dict['lbw'] = lbw
            home_dict['ubw'] = ubw
            home_dict['traj_full'] = traj_full
            
        mpc_peak = MPC_peak_state(N-1, len(home_list))

        p_num = mpc_peak.p(0)
        p_num['weight'] = 20

        w0 = mpc_peak.w(0)
        lbw = mpc_peak.w(-inf)
        ubw = mpc_peak.w(inf)
        lbw['peak', :] = 0
        ubw['peak', :] = mpc_peak.n_homes * mpc_peak.P_max

        x = mpc_peak.w(0)
        x['peak', :] = state_0['peak']

        traj_full = np.zeros(T)
        traj_full[0] = state_0['peak']

        state_dict['peak']['mpc'] = mpc_peak
        state_dict['peak']['x'] = x
        state_dict['peak']['p_num'] = p_num
        state_dict['peak']['w0'] = w0
        state_dict['peak']['lbw'] = lbw
        state_dict['peak']['ubw'] = ubw
        state_dict['peak']['traj_full'] = traj_full
        
        state_dict['dual_variables'] = {
            'current_value': np.ones(N),
            'traj_full': np.ones((T,N))
        }
        
        return state_dict
    
    def run_full(self):
        mpc_single_home = self.state_dict['mpc_single_home']
        for home in self.state_dict['homes'].values():
            mpc_single_home.prepare_MPC_action(
                home['w0'], home['x'], self.N, home['lbw'], home['ubw']
            )

        w0_list = [self.state_dict['homes'][home]['w0'].master for home in self.home_list]
        x_list = [self.state_dict['homes'][home]['x'].master for home in self.home_list]
        lbw_list = [self.state_dict['homes'][home]['lbw'].master for home in self.home_list]
        ubw_list = [self.state_dict['homes'][home]['ubw'].master for home in self.home_list]
        p_num_list = [self.state_dict['homes'][home]['p_num'].master for home in self.home_list]
        
        m = self.executor.map(mpc_single_home.get_MPC_action, 
                              w0_list, lbw_list, ubw_list, p_num_list, chunksize=8)
        
        return list(m)[0]
        

#%%

def price_func_exp(x): # Function to emulate fluctuating power prices at 5 minute intervals
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))

N = 288 # MPC horizon (how far it optimizes)
T = 10 # Running time (how many times do we solve opt. prob.)
spot_prices = np.fromfunction(price_func_exp, (N+T,)) # Spot prices for two days, 5 min intervals
outdoor_temperature = 10 * np.ones(N)
home_list = ['axel', 'seb']
ref_temp = {'axel': 21*np.ones(N), 'seb': 24*np.ones(N)}

state_0_axel = {'wall': 9, 'room': 10, 'P_hp': 0}
state_0_seb = {'wall': 14, 'room': 16, 'P_hp': 0}
# state_0_kang = {'wall': 24, 'room': 28, 'P_hp': 0}
state_0 = {'axel': state_0_axel, 'seb': state_0_seb, 'peak': 0}





#%%
    
if __name__ == '__main__':
    dmpc = DMPC(N, T, home_list, state_0, spot_prices, outdoor_temperature)
    res = dmpc.run_full()
    w = dmpc.state_dict['mpc_single_home'].w(res)