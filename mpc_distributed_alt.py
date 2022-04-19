#%%
from importlib.abc import MetaPathFinder
from casadi import *
from casadi.tools import *
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from time import time 
from pickle import dumps, loads
from concurrent.futures import ProcessPoolExecutor

class MPC_single_home():
    def __init__(self, N, P_max = 1.5):
        self.N = N
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
        print(f'home action PID: {os.getpid()}')
        return solution['x']
    
    @staticmethod
    def update_initial_state(w0, x_opt, N):
        w0['state', :N-1] = x_opt['state', 1:]
        w0['state', -1] = x_opt['state', -1]
        
        w0['input', :N-2] = x_opt['input', 1:]
        w0['input', -1] = x_opt['input', -1]
        
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        lbw['state', 0, 'wall'] = w0['state', 0, 'wall']
        ubw['state', 0, 'wall'] = w0['state', 0, 'wall']
        
        lbw['state', 0, 'room'] = w0['state', 0, 'room']
        ubw['state', 0, 'room'] = w0['state', 0, 'room']
        
        lbw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        ubw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        
    @staticmethod   
    def prepare_MPC_action(w0, x_opt, N, lbw, ubw):
        MPC_single_home.update_initial_state(w0, x_opt, N)
        MPC_single_home.update_constraints(w0, lbw, ubw)

class DMPC_constant():
    def __init__(self, N, T, home_list, state_0, spot_prices, outdoor_temperature, reference_temperature, dual_variable, max_total_power):
        self.N = N
        self.T = T
        self.home_list = home_list
        self.spot_prices = spot_prices
        self.outdoor_temperature = outdoor_temperature
        self.reference_temperature = reference_temperature
        self.max_total_power = max_total_power
        self.state_dict = self.build_state_dict(
            N, T, home_list, state_0, spot_prices, outdoor_temperature, reference_temperature, dual_variable
            )
        self.executor = ProcessPoolExecutor()
        
    @staticmethod
    def build_state_dict(N, T, home_list, state_0, spot_prices, outdoor_temperature, reference_temperature, dual_variable):
        mpc_single_home = MPC_single_home(N)
        state_dict = {'mpc_single_home': mpc_single_home, 'homes': dict.fromkeys(home_list)}
        for home in home_list:
            home_dict = {}
            
            p_num = mpc_single_home.p(0)
            p_num['outdoor_temperature', :] = list(outdoor_temperature[:N])
            p_num['reference_temperature', :] = list(reference_temperature[home][:N])
            p_num['spot_price', :] = list(spot_prices[:N-1])
            p_num['dual_variable', :] = list(dual_variable)
            p_num['weights', 'energy'] = 100
            p_num['weights', 'comfort'] = 1
            p_num['model', 'rho_out'] = 0.18
            p_num['model', 'rho_in'] = 0.37
            p_num['model', 'COP'] = 3.5
            
            w0 = mpc_single_home.w(0)
            w0['state', :, 'room'] = state_0[home]['room']
            w0['state', :, 'wall'] = state_0[home]['wall']
            w0['input', :, 'P_hp'] = state_0[home]['P_hp']
            lbw = mpc_single_home.w(-inf)
            ubw = mpc_single_home.w(inf)
            lbw['input', :, 'P_hp'] = 0
            ubw['input', :, 'P_hp'] = mpc_single_home.P_max
            
            x = mpc_single_home.w(0)
            
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
            state_dict['homes'][home] = home_dict
        
        
        state_dict['dual_variable'] = {
            'current_value': dual_variable, #np.ones(N-1),
            'traj_full': np.ones((T,N-1))
        }
        state_dict['dual_variable']['traj_full'][0] = dual_variable
        
        return state_dict
    
    def update_home_params(self, t):
        for home, home_dict in zip(self.home_list, self.state_dict['homes'].values()):
            home_dict['p_num']['outdoor_temperature', :] = list(self.outdoor_temperature[t:t+N])
            home_dict['p_num']['reference_temperature', :] = list(self.reference_temperature[home][t:t+N])
            home_dict['p_num']['spot_price', :] = list(self.spot_prices[t:t+N-1])
    
    def update_state(self, result_list):
        for res_home, home_dict in zip(result_list, self.state_dict['homes'].values()):
            home_dict['x'] = res_home
            
    def update_w0(self, result_list):
        for res_home, home_dict in zip(result_list, self.state_dict['homes'].values()):
            home_dict['w0'] = res_home
    
    def update_state_trajectory(self, result_list, t):
        for res_home, home_dict in zip(result_list, self.state_dict['homes'].values()):
            home_dict['traj_full']['room'][t] = res_home['state', 0, 'room']
            home_dict['traj_full']['wall'][t] = res_home['state', 0, 'wall']
            home_dict['traj_full']['P_hp'][t] = res_home['input', 0, 'P_hp']
    
    def update_dual_variable_trajectory(self, t):
        self.state_dict['dual_variable']['traj_full'][t] = self.state_dict['dual_variable']['current_value']
    
    def run_full(self):
        
        mpc_single_home = self.state_dict['mpc_single_home']
        
        for t in range(1, self.T):
            
            for home in self.state_dict['homes'].values():
                mpc_single_home.update_constraints(
                    home['w0'], home['lbw'], home['ubw']
                )
            # self.update_dual_variable(t)
            result_list = self.repeat_single_step()
            return result_list
        
            for home in self.state_dict['homes'].values():
                mpc_single_home.prepare_MPC_action(
                    home['w0'], home['x'], self.N, home['lbw'], home['ubw']
                )

            w0_list = [self.state_dict['homes'][home]['w0'].master for home in self.home_list]
            lbw_list = [self.state_dict['homes'][home]['lbw'].master for home in self.home_list]
            ubw_list = [self.state_dict['homes'][home]['ubw'].master for home in self.home_list]
            p_num_list = [self.state_dict['homes'][home]['p_num'].master for home in self.home_list]
            
            result_map = self.executor.map(mpc_single_home.get_MPC_action, 
                                w0_list, lbw_list, ubw_list, p_num_list, chunksize=8)
            result_list = [self.state_dict['mpc_single_home'].w(x) for x in list(result_map)]

            self.update_state_trajectory(result_list, t)
                
            
            self.update_home_params(t)
            
            print(f'Iteration {t} / {self.T}')
            
            
    
    def update_current_dual_variable(self, alpha):
        power_sum = np.zeros(self.N-1)
        for home_dict in self.state_dict['homes'].values():
            power_sum += np.array(vertcat(*home_dict['x']['input',:,'P_hp'])).flatten()
        # alpha = 20
        self.state_dict['dual_variable']['current_value'] += alpha * (power_sum - self.max_total_power * np.ones(self.N-1))
        self.state_dict['dual_variable']['current_value'][self.state_dict['dual_variable']['current_value'] < 0] = 0 #round up negative numbers to 0
        
        for home_dict in self.state_dict['homes'].values():
            home_dict['p_num']['dual_variable'] = list(self.state_dict['dual_variable']['current_value']) 

        
    def repeat_single_step(self):
        mpc_single_home = self.state_dict['mpc_single_home']
        
        lbw_list = [self.state_dict['homes'][home]['lbw'].master for home in self.home_list]
        ubw_list = [self.state_dict['homes'][home]['ubw'].master for home in self.home_list]
        p_num_list = [self.state_dict['homes'][home]['p_num'].master for home in self.home_list]
        
        numIt = 0
        while True:
            
            
            dual_variable_last = copy(self.state_dict['dual_variable']['current_value'])
            iteration_difference = 1
            

            w0_list = [self.state_dict['homes'][home]['w0'].master for home in self.home_list]
            
            result_map = self.executor.map(mpc_single_home.get_MPC_action, 
                                w0_list, lbw_list, ubw_list, p_num_list, chunksize=8)
            result_list = [self.state_dict['mpc_single_home'].w(x) for x in list(result_map)]

            # self.update_state(result_list)
            self.update_w0(result_list)
            self.update_state(result_list)
                
            alpha = 20 / sqrt(1+numIt)
            self.update_current_dual_variable(alpha)
            self.update_dual_variable_trajectory(numIt+1)

            iteration_difference = (np.abs(self.state_dict['dual_variable']['current_value'] - dual_variable_last)).mean()
            avg_value = np.average(self.state_dict['dual_variable']['current_value'])
            print(f'Internal iteration number {numIt}, iteration difference = {round(iteration_difference,5)}, average dual variable value: {round(avg_value,2)}')
            if iteration_difference < 0.01 or numIt >= 30:
                return result_list
            
            numIt += 1

#%%

def price_func_exp(x): # Function to emulate fluctuating power prices at 5 minute intervals
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))
N = 288 # MPC horizon (how far it optimizes)
T = 70 # Running time (how many times do we solve opt. prob.)
spot_prices = np.fromfunction(price_func_exp, (N+T,)) # Spot prices for two days, 5 min intervals
outdoor_temperature = 10 * np.ones(N+T)
home_list = ['axel', 'seb']
ref_temp = {'axel': 21*np.ones(N+T), 'seb': 24*np.ones(N+T)} # Desired temperature for each home
state_0_axel = {'wall': 9, 'room': 10, 'P_hp': 0}
state_0_seb = {'wall': 14, 'room': 16, 'P_hp': 0}
# state_0_kang = {'wall': 24, 'room': 28, 'P_hp': 0}
state_0 = {'axel': state_0_axel, 'seb': state_0_seb}
dual_variable = 5 * np.ones(N-1) # A variable for the power output of each time step
# dual_variable[1]= 300
max_total_power = len(home_list) * 0.75

#%%
    
if __name__ == '__main__':
    dmpc = DMPC_constant(
        N, T, home_list, state_0, spot_prices, outdoor_temperature, ref_temp, dual_variable, max_total_power
        )
    result_list = dmpc.run_full()
    print(dmpc.state_dict['dual_variable']['traj_full'][:15,:5])
    axel = result_list[0]
    seb = result_list[1]
    axel = np.array(vertcat(*axel['input', :, 'P_hp'])).flatten()
    seb = np.array(vertcat(*seb['input', :, 'P_hp'])).flatten()
    figa, axa = plt.subplots()
    axa.plot(axel, label='axel')
    axa.plot(seb, label='seb')
    axa.set_title('Total power consumption [kW]')
    axa.set_xlabel('Time step (5 minute intervals)')
    axa.legend()
    plt.show()
    
    # %%
    traj_dv = dmpc.state_dict['dual_variable']['traj_full'][:30]
    x, y = np.meshgrid(np.arange(traj_dv.shape[1]), np.arange(traj_dv.shape[0]))
    z = traj_dv
    figdv, axdv = plt.subplots(subplot_kw={"projection": "3d"})
    axdv.set_title('Dual variable values')
    surf = axdv.plot_surface(x, y, z)
    axdv.zaxis.set_rotate_label(False)
    axdv.set_xlabel('MPC horizon step')
    axdv.set_ylabel('Iteration')
    axdv.set_zlabel('$\lambda$')
    plt.show()
    
    '''
    figh, axh = plt.subplots()
    for home, home_dict in zip(dmpc.home_list, dmpc.state_dict['homes'].values()):
        axh.plot(home_dict['traj_full']['wall'], label=home)
    axh.legend()
    axh.set_title('Wall temperature [°C]')
    
    figr, axr = plt.subplots()
    for home, home_dict in zip(dmpc.home_list, dmpc.state_dict['homes'].values()):
        axr.plot(home_dict['traj_full']['room'], label=home)
    axr.legend()
    axr.set_title('Room temperature [°C]')
    
    figpw, axpw = plt.subplots()
    for home, home_dict in zip(dmpc.home_list, dmpc.state_dict['homes'].values()):
        axpw.plot(home_dict['traj_full']['P_hp'], label=home)
    axpw.legend()
    axpw.set_title('Heat pump power [W]')

    
    plt.show()
    
    '''


#%%
