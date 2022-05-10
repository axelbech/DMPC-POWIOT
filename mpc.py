from copy import copy
from pickle import dumps
import os
import time

from weakref import ref
from casadi import *
from casadi.tools import *
import numpy as np


class MPC():
    """Model predictive control template for model predictive control using 
    CasADi symbolics and IPOPT for optimization
    
    """
    def __init__(self, N: int, T: int, name: str, params: dict):
        """Create an MPC object

        Args:
            N (int): optimization window length
            T (int): simulation time
            name (str): instance name for parameter indexing
            params (dict): initial states, bounds, optimization parameters
        """
        self.N = N
        self.T = T
        self.name = name
        self.params = params

        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.w0 = self.get_initial_state()
        self.lbw, self.ubw = self.get_state_bounds()
        self.p_num = self.get_numerical_parameters()
        self.w_opt = copy(self.w)(0)
        
        self.solver = self.get_solver()
        
        self.traj_full = self.get_trajectory_structure()
        
        dumps(self) # Another quick fix for serializability
        
    def get_decision_variables(self):
        """Builds the decision variable
        
        Returns:
            w (struct_symMX): decision variable
        """
        pass
    
    def get_parameters_structure(self):
        """Builds the the parameters structure
        
        Returns: 
            p (struct_symMX): parameter structure
        """
        pass
    
    def get_initial_state(self):
        """Builds initial decision variable state from params
        
        Returns:
            w0 (DMStruct): initial state for optimization
        """
        pass
    
    def get_state_bounds(self):
        """Builds bounds on the state w
        
        Returns:
            lbw (DMStruct): lower bounds on w
            ubw (DMStruct): upper bounds on w
        """
        pass
    
    def get_numerical_parameters(self):
        """Builds numerical optimization parameters from params
        
        Returns:
            p_num (DMStruct): numerical parameters for optimization
        """
        pass
    
    def get_cost_funtion(self):
        """Builds the cost function
        
        Returns:
            J (MX): cost function
        """
        pass
    
    def get_constraint_functions(self):
        """Builds constraint functions and bounds
        
        Returns:
            g (list(MX)): constraint functions
            lbg (list): lower bounds on g
            ubg (list): upper bounds on g
        """
        pass
    
    def get_trajectory_structure(self):
        """Builds trajectory structure as dictionary of lists
        
        Returns:
            traj_full (dict): full trajectory
        """
        pass
    
    def get_solver(self):
        """Builds casadi non linear programming solver
        
        Returns:
            solver (casadi::Function): NLP solver for opt. problem
        """
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_function(), 
            'x': self.w, 
            'g': vertcat(*(g)), 
            'p': self.p
            }
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def solve_optimization(self):
        """Completes optimization for one MPC step

        Returns:
            x (list): optimized state variables
            f (list): optimal cost function value
        """
        solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.p_num
            )
        assert self.solver.stats()['success']
        # print(f'home action PID: {os.getpid()}')
        return solution['x'], solution['f']
    
    def set_optimal_state(self, w_res):
        """Set internal optimal state with result from optimization
        
        """
        if not isinstance(w_res, structure3.DMStruct):
            w_res = self.w(w_res)
        self.w_opt = w_res
        
    def set_initial_state(self, w_res):
        """Set internal initial state
        
        """
        if not isinstance(w_res, structure3.DMStruct):
            w_res = self.w(w_res)
        self.w0 = w_res
    
    def update_initial_state(self):
        """Using w_opt from time t, update the start state w0 for time t+1
        
        """
        pass
    
    def update_constraints(self):
        """Update the constraints lbw and ubw so they ensure the start state in
        w0 at the first horizon step does not change during optimization. The 
        first input in w0 may change
        
        """
        pass
    
    def update_trajectory(self):
        """Update trajectory with w_opt"""
        pass
    
    def update_parameters_generic(self, **kwargs):
        """Modulary updates parameters, assuming they are already of the right
        size.
        
        """
        for key, value in kwargs.items():
            self.p_num[key] = list(value) 
            
    def update_parameters(self, t):
        """Update time variant optimization parameters from params

        Args:
            p_ext (dict): external parameters structure
            t (int): time step
        """
        pass
    
    def run_full(self, return_dict):
        """Run the MPC simulation for T time steps

        Args:
            return_dict (dict): for returning the results
        """
        
        for t in range(self.T):
            print(f'{self.name}: time step {t}, pid = {os.getpid()}')
            
            self.update_parameters(t) # prepare mpc params
            
            self.update_constraints() # prepare mpc start constraints
                
            w_opt, f_opt = self.solve_optimization()
                
            self.set_optimal_state(w_opt)
            
            self.update_trajectory()
            
            self.update_initial_state()
        
        return_dict['traj_full'] = self.traj_full
        return_dict['params'] = self.params


class MPCDistributed(MPC):
    
    def run_full(self, return_dict, public_coordination, private_coordination):
        """Run the distributed MPC simulation with coordination from a 
        coordinator object

        Args:
            return_dict (dict): for returning the results
            coordination_dict (dict): to coordinate time and get dual variable
            dv_contribution (list): for sending dual variable update contributions
        """
        t = 0
        while t < self.T:

            self.update_parameters(t) # prepare mpc params
            
            self.update_constraints() # prepare mpc start constraints
            
            while t == public_coordination['t']:
                while not private_coordination['f_opt'] == None:
                    time.sleep(0.05)
                if not t == public_coordination['t']:
                    break
                self.update_parameters_generic(
                    dual_variables=public_coordination['dual_variables']
                    )
                w_opt, f_opt = self.solve_optimization()
                self.set_optimal_state(self.w(w_opt))
                # self.set_initial_state(self.w(w_opt))
                private_coordination['dual_update_contribution'] = \
                    self.get_dual_update_contribution()
                private_coordination['f_opt'] = f_opt
                print(f'{self.name}: time step {t}, pid = {os.getpid()}, w_opt = {self.w_opt.master[0]}')
            
            self.update_trajectory()
            
            self.update_initial_state()
            
            t = public_coordination['t']

        return_dict['traj_full'] = self.traj_full
        return_dict['params'] = self.params
        
    def get_dual_update_contribution(self):
        """Get the controllers contribution to the dual variable update

        Returns:
            dv_contribution (list): dual variable update contribution
        """
        pass


class MPCSingleHome(MPC):
        
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1)
            ])
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall_temp')
        room = MX.sym('room_temp')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function(
            'wall_plus', 
            [rho_out, rho_in, wall, room, OutTemp],
            [wall_plus]
            )

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function(
            'room_plus',
            [rho_in, room, wall, COP, Pow],
            [room_plus]
            )

        return wall_func, room_func

    def get_decision_variables(self):
        MPC_states = struct_symMX([
            entry('room_temp'), 
            entry('wall_temp'),
            entry('slack_temp')
            ])
        MPC_inputs = struct_symMX([entry('P_hp')])

        w = struct_symMX([
        entry('state', struct=MPC_states, repeat=self.N),
        entry('input', struct=MPC_inputs, repeat=self.N - 1)
        ])
        return w
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)
        
        w0['state',:,'room_temp'] = self.params['initial_state']['room_temp']
        w0['state',:,'wall_temp'] = self.params['initial_state']['wall_temp']
        
        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        
        lbw['input',:,'P_hp'] = 0
        ubw['input',:,'P_hp'] = self.params['bounds']['P_max']
        
        lbw['state',:,'slack_temp'] = 0
        
        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        
        opt_params = self.params['opt_params']
        p_num['energy_weight'] = opt_params['energy_weight']
        p_num['comfort_weight'] = opt_params['comfort_weight']
        p_num['slack_min_weight'] = opt_params['slack_min_weight']
        p_num['rho_out'] = opt_params['rho_out']
        p_num['rho_in'] = opt_params['rho_in']
        p_num['COP'] = opt_params['COP']
        
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['room_temp'] = []
        traj_full['wall_temp'] = []
        traj_full['P_hp'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
        for k in range(self.N-1): 
            J += self.p['comfort_weight'] * \
            (self.w['state', k+1, 'room_temp'] - self.p['reference_temperature',k+1])**2
            
            J += self.p['slack_min_weight']*self.w['state',k+1,'slack_temp']**2
        
            J += self.p['energy_weight'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
        # for k in range(self.N - 1): # Input not defined for the last timestep
        
        return J
    
    def get_constraint_functions(self):
        rho_out = self.p['rho_out']
        rho_in = self.p['rho_in']
        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for k in range(self.N - 1):
            wall = self.w['state', k, 'wall_temp']
            room = self.w['state', k, 'room_temp']
            Pow = self.w['input', k, 'P_hp']
            out_temp = self.p['outdoor_temperature', k]
            
            wall_plus = self.wall_func(
                rho_out, 
                rho_in, 
                wall, 
                room, 
                out_temp
                )
            room_plus = self.room_func(
                rho_in,
                room,
                wall,
                self.p['COP'],
                Pow
                )
            
            g.append(wall_plus - self.w['state', k+1, 'wall_temp'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room_temp'])
            lbg.append(0)
            ubg.append(0)
            
            g.append(self.p['min_temperature', k] 
                     - self.w['state', k, 'room_temp'] 
                     - self.w['state', k, 'slack_temp'])
            lbg.append(-inf)
            ubg.append(0)
            
            g.append
                
        return g, lbg, ubg
    
    def update_trajectory(self):
        self.traj_full['room_temp'].append(self.w_opt['state',0,'room_temp'].__float__())
        self.traj_full['wall_temp'].append(self.w_opt['state',0,'wall_temp'].__float__())
        self.traj_full['P_hp'].append(self.w_opt['input',0,'P_hp'].__float__())

    def update_initial_state(self):
        self.w0['state', :self.N-1] = self.w_opt['state', 1:]
        self.w0['state', -1] = self.w_opt['state', -1]
        
        self.w0['input', :self.N-2] = self.w_opt['input', 1:]
        self.w0['input', -1] = self.w_opt['input', -1]
        
    def update_constraints(self):
        self.lbw['state', 0, 'wall_temp'] = self.w0['state', 0, 'wall_temp']
        self.ubw['state', 0, 'wall_temp'] = self.w0['state', 0, 'wall_temp']
        
        self.lbw['state', 0, 'room_temp'] = self.w0['state', 0, 'room_temp']
        self.ubw['state', 0, 'room_temp'] = self.w0['state', 0, 'room_temp']
        
    def update_parameters(self, t):
        opt_params = self.params['opt_params']
        self.p_num['outdoor_temperature'] = (
            opt_params['outdoor_temperature'][t:t+self.N]
        )
        self.p_num['reference_temperature'] = (
            opt_params['reference_temperature'][t:t+self.N]
        )
        self.p_num['min_temperature'] = (
            opt_params['min_temperature'][t:t+self.N]
        )
        self.p_num['spot_price'] = (
            opt_params['spot_price'][t:t+self.N-1]
        )
        self.p_num['ext_power'] = (
            opt_params['ext_power_avg'][t:t+self.N-1]
        )
        self.p_num['ext_power', 0] = (
            opt_params['ext_power_real'][t]
        )
        return


class MPCSingleHomePeak(MPC):
        
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('peak_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1)
            ])
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall_temp')
        room = MX.sym('room_temp')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function(
            'wall_plus', 
            [rho_out, rho_in, wall, room, OutTemp],
            [wall_plus]
            )

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function(
            'room_plus',
            [rho_in, room, wall, COP, Pow],
            [room_plus]
            )

        return wall_func, room_func

    def get_decision_variables(self):
        return struct_symMX([
                entry('room_temp', repeat=self.N),
                entry('wall_temp', repeat=self.N),
                entry('slack_temp', repeat=self.N),
                entry('peak_state', repeat=self.N),
                entry('P_hp', repeat=self.N-1),
            ])
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)
        w0['room_temp',:] = self.params['initial_state']['room_temp']
        w0['wall_temp',:] = self.params['initial_state']['wall_temp']
        w0['peak_state',:] = self.params['initial_state']['peak_state']
        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        lbw['P_hp', :] = 0
        ubw['P_hp', :] = self.params['bounds']['P_max']
        lbw['slack_temp', :] = 0
        lbw['peak_state', :] = 0
        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        opt_params = self.params['opt_params']
        p_num['energy_weight'] = opt_params['energy_weight']
        p_num['comfort_weight'] = opt_params['comfort_weight']
        p_num['slack_min_weight'] = opt_params['slack_min_weight']
        p_num['peak_weight'] = opt_params['peak_weight']
        p_num['rho_out'] = opt_params['rho_out']
        p_num['rho_in'] = opt_params['rho_in']
        p_num['COP'] = opt_params['COP']
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['room_temp'] = []
        traj_full['wall_temp'] = []
        traj_full['P_hp'] = []
        traj_full['peak_state'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
        
#         for k in range(self.N):
#             J += self.p['comfort_weight'] * \
# (self.w['room_temp', k] - self.p['reference_temperature', k])**2
#             J += self.p['slack_min_weight']*self.w['slack_temp',k]**2

        for k in range(self.N-1): # Dont penalize fixed first state
            J += self.p['comfort_weight'] * \
(self.w['room_temp', k+1] - self.p['reference_temperature', k+1])**2
            J += self.p['slack_min_weight']*self.w['slack_temp',k+1]**2
            
            # J += self.p['peak_weight'] * self.w['peak_state', k+1]**2 # Might actually be INCORRECT
            J += self.p['peak_weight'] * self.w['peak_state', k+1]
            
            J += self.p['energy_weight'] * \
                self.p['spot_price', k] * self.w['P_hp', k]
                
        # J += self.p['peak_weight'] * self.N-1 * self.w['peak_state', -1] # LAST STATE!

        return J
    
    def get_constraint_functions(self):

        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for k in range(self.N - 1):
            rho_out = self.p['rho_out']
            rho_in = self.p['rho_in']
            wall = self.w['wall_temp', k]
            room = self.w['room_temp', k]
            Pow = self.w['P_hp', k]
            out_temp = self.p['outdoor_temperature', k]
            COP = self.p['COP']
            
            wall_plus = self.wall_func(
                rho_out, 
                rho_in, 
                wall, 
                room, 
                out_temp
                )
            room_plus = self.room_func(
                rho_in,
                room,
                wall,
                COP,
                Pow
                )
            
            g.append(wall_plus - self.w['wall_temp', k+1])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['room_temp', k+1])
            lbg.append(0)
            ubg.append(0)
            
            g.append(self.p['min_temperature', k] 
                    - self.w['room_temp', k] 
                    - self.w['slack_temp', k])
            lbg.append(-inf)
            ubg.append(0)
            
        for k in range(self.N - 1): # This goes away when dualized
            g.append(self.w['peak_state', k+1] - self.w['peak_state', k])
            lbg.append(0)
            ubg.append(inf)
        for k in range(self.N - 1):
            power_sum = 0
            power_sum += self.w['P_hp', k]
            power_sum += self.p['ext_power', k]
            g.append(self.w['peak_state', k+1] - power_sum) # peak state must be greater than sum of power at k
            lbg.append(0)
            ubg.append(inf)
                
        return g, lbg, ubg
    
    def update_trajectory(self):
        self.traj_full['room_temp'].append(self.w_opt['room_temp',0].__float__())
        self.traj_full['wall_temp'].append(self.w_opt['wall_temp',0].__float__())
        self.traj_full['P_hp'] .append(self.w_opt['P_hp',0].__float__())
        self.traj_full['peak_state'].append(self.w_opt['peak_state',:0].__float__())

    def update_initial_state(self):
        self.w0['room_temp',:self.N-1]=self.w_opt['room_temp',1:]
        self.w0['room_temp',-1]=self.w_opt['room_temp', -1]
        
        self.w0['wall_temp',:self.N-1]=self.w_opt['wall_temp',1:]
        self.w0['wall_temp',-1]=self.w_opt['wall_temp', -1]
        
        self.w0['peak_state', :self.N-1] = self.w_opt['peak_state', 1:]
        self.w0['peak_state', -1] = self.w_opt['peak_state', -1]
        
        self.w0['P_hp',:self.N-2]=self.w_opt['P_hp',1:]
        self.w0['P_hp',-1]=self.w_opt['P_hp', -1]
            
    def update_constraints(self):
        self.lbw['wall_temp', 0] = self.w0['wall_temp', 0]
        self.ubw['wall_temp', 0] = self.w0['wall_temp', 0]
        
        self.lbw['room_temp', 0] = self.w0['room_temp', 0]
        self.ubw['room_temp', 0] = self.w0['room_temp', 0]
        
        self.lbw['peak_state', 0] = self.w0['peak_state', 0]
        self.ubw['peak_state', 0] = self.w0['peak_state', 0]
            
        # self.lbw['peak_state', 0] = round(float(self.w_opt['peak_state',0]), 6)
                   
    def update_parameters(self, t):
        opt_params = self.params['opt_params']
        self.p_num['outdoor_temperature'] = (
            opt_params['outdoor_temperature'][t:t+self.N]
        )
        self.p_num['reference_temperature'] = (
            opt_params['reference_temperature'][t:t+self.N]
        )
        self.p_num['min_temperature'] = (
        opt_params['min_temperature'][t:t+self.N]
        )
        self.p_num['spot_price'] = (
            opt_params['spot_price'][t:t+self.N-1]
        )
        self.p_num['ext_power'] = (
            opt_params['ext_power_avg'][t:t+self.N-1]
        )
        self.p_num['ext_power', 0] = (
            opt_params['ext_power_real'][t]
        )
        return


class MPCSingleHomePeakDistributed(MPCDistributed, MPCSingleHomePeak):
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('peak_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1),
                entry('dual_variables', repeat=self.N-1)
            ])
        
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['room_temp'] = []
        traj_full['wall_temp'] = []
        traj_full['P_hp'] = []
        traj_full['peak_state'] = []
        traj_full['dualized_constraints_values'] = []
        return traj_full
        
    def get_cost_function(self):
        J = super().get_cost_function()
        
        for k in range(self.N-1):
            J += self.p['dual_variables',k] * (
                self.w['peak_state',k] - self.w['peak_state', k+1]
            )
            
        return J
    
    def get_constraint_functions(self):
        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for k in range(self.N - 1):
            rho_out = self.p['rho_out']
            rho_in = self.p['rho_in']
            wall = self.w['wall_temp', k]
            room = self.w['room_temp', k]
            Pow = self.w['P_hp', k]
            out_temp = self.p['outdoor_temperature', k]
            COP = self.p['COP']
            
            wall_plus = self.wall_func(
                rho_out, 
                rho_in, 
                wall, 
                room, 
                out_temp
                )
            room_plus = self.room_func(
                rho_in,
                room,
                wall,
                COP,
                Pow
                )
            
            g.append(wall_plus - self.w['wall_temp', k+1])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['room_temp', k+1])
            lbg.append(0)
            ubg.append(0)
            
            g.append(self.p['min_temperature', k] 
                    - self.w['room_temp', k] 
                    - self.w['slack_temp', k])
            lbg.append(-inf)
            ubg.append(0)
            
        # for k in range(self.N - 2): # This goes away when dualized
        #     g.append(self.w['peak_state', k+1] - self.w['peak_state', k])
        #     lbg.append(0)
        #     ubg.append(inf)
        for k in range(self.N - 1):
            power_sum = 0
            power_sum += self.w['P_hp', k]
            power_sum += self.p['ext_power', k]
            g.append(self.w['peak_state', k+1] - power_sum) # peak state must be greater than sum of power at k
            lbg.append(0)
            ubg.append(inf)
                
        return g, lbg, ubg

    def get_dual_update_contribution(self):
        peak_state_next=np.array(vertcat(*self.w_opt['peak_state',1:])).flatten()
        peak_state_current = np.array(vertcat(*self.w_opt['peak_state', :-1])).flatten()
        dual_update_contribution = peak_state_current - peak_state_next # Flipped for h(x)<=0 constraint standard
        self.traj_full['dualized_constraints_values'].append(dual_update_contribution.tolist())
        return dual_update_contribution


class MPCSingleHomeDistributed(MPCDistributed, MPCSingleHome):
        
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1),
                entry('dual_variables', repeat=self.N-1)
            ])
    
    def get_cost_function(self):
        J = 0
        for k in range(self.N-1): 
            J += self.p['comfort_weight'] * \
            (self.w['state', k+1, 'room_temp'] - self.p['reference_temperature',k+1])**2
            
            J += self.p['slack_min_weight']*self.w['state',k+1,'slack_temp']**2
        
            J += self.p['energy_weight'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
                
            J +=  self.p['dual_variables', k] * self.w['input', k, 'P_hp']
        # for k in range(self.N - 1): # Input not defined for the last timestep
            # J += self.p['energy_weight'] * self.p['dual_variables', k] *\
            #     self.w['input', k, 'P_hp'] # From dual decomposition, dual_var like a power price
        
        return J
    
    def get_dual_update_contribution(self):
        heat_pump_power=np.array(vertcat(*self.w_opt['input',:, 'P_hp'])).flatten()
        ext_power = np.array(vertcat(*self.p_num['ext_power', :])).flatten()
        total_power = heat_pump_power + ext_power
        # print(self.name, self.p_num['dual_variables',0], total_power[:5])
        return total_power


class MPCPeakStateDistributed(MPCDistributed):
        
    def get_decision_variables(self):
        return struct_symMX([entry('peak_state', repeat=self.N-1)])
    
    def get_parameters_structure(self):
        return struct_symMX([
            entry('peak_weight'), 
            entry('dual_variables', repeat=self.N-1),
            # entry('ext_power', repeat=self.N-1) # QUICK FIX!
            ])

    def get_cost_function(self):
        J = 0
        for k in range(self.N-1):
            # J -= self.p['peak_weight'] * \
            #     self.p['dual_variables', k] * self.w['peak_state', k]
            J -= self.p['dual_variables', k] * self.w['peak_state', k]
            J += self.p['peak_weight'] * self.w['peak_state', k]
        return J
            
    def get_constraint_functions(self):
        g = []
        lbg = []
        ubg = []
        for k in range(self.N - 2):
            g.append(self.w['peak_state', k+1] - self.w['peak_state', k])
            lbg.append(0)
            ubg.append(inf)
        return g, lbg, ubg
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)

        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)

        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        p_num['peak_weight'] = self.params['opt_params']['peak_weight']
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['peak_state'] = []
        return traj_full
    
    def get_dual_update_contribution(self):
        # print(self.name, self.p_num['dual_variables', 0], -np.array(vertcat(*self.w_opt['peak_state',:])).flatten()[:5])
        return -np.array(vertcat(*self.w_opt['peak_state',:])).flatten()
    
    def update_trajectory(self):
        self.traj_full['peak_state'].append(
            round(self.w_opt['peak_state',0].__float__(), 6)
            )
    
    def update_initial_state(self):
        self.w0['peak_state', :self.N-2] = self.w_opt['peak_state', 1:]
        self.w0['peak_state', -1] = self.w_opt['peak_state', -1]
        
    def update_constraints(self):
        # Need numerical tweaking to ensure feasability
        # self.lbw['peak_state'] = self.p_num['ext_power']
        self.lbw['peak_state', 0] = round(float(self.w_opt['peak_state',0]), 6)
        # if float(self.p_num['ext_power',0]) >= round(float(self.w_opt['peak_state',0]), 6):
        #     self.lbw['peak_state', 0] = self.p_num['ext_power', 0]
        
    # def update_parameters(self, t): # ANOTHER QUICK FIX
    #     opt_params = self.params['opt_params']
    #     self.p_num['ext_power'] = (
    #         opt_params['ext_power_avg'][t:t+self.N-1]
    #     )
    #     self.p_num['ext_power', 0] = (
    #         opt_params['ext_power_real'][t]
    #     )
    #     return


class MPCPeakStateDistributedQuadratic(MPCPeakStateDistributed):
    def get_cost_function(self):
        J = 0
        for k in range(self.N-1):
            # J -= self.p['peak_weight'] * \
            #     self.p['dual_variables', k] * self.w['peak_state', k]
            J -= self.p['dual_variables', k] * self.w['peak_state', k]
            J += self.p['peak_weight'] * self.w['peak_state', k]**2
        return J


class MPCCentralizedHomePeak(MPC):
    
    def __init__(self, N: int, T: int, name: str, params: dict):
        homes = list(params.keys())
        if 'peak' in homes: homes.remove('peak')
        self.homes = homes
        super().__init__(N, T, name, params)
        
    def get_parameters_structure(self):
        param_list = []
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1)
            ])
            param_list.append(entry(home, struct=home_struct))
            
        peak_struct = struct_symMX([
            entry('peak_weight')
            ])
        param_list.append(entry('peak_state', struct=peak_struct))
        
        p = struct_symMX(param_list)
        
        return p
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall_temp')
        room = MX.sym('room_temp')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function(
            'wall_plus', 
            [rho_out, rho_in, wall, room, OutTemp],
            [wall_plus]
            )

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function(
            'room_plus',
            [rho_in, room, wall, COP, Pow],
            [room_plus]
            )

        return wall_func, room_func

    def get_decision_variables(self):
        state_list = []
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('room_temp', repeat=self.N),
                entry('wall_temp', repeat=self.N),
                entry('slack_temp', repeat=self.N),
                entry('P_hp', repeat=self.N-1)
            ])
            state_list.append(entry(home, struct=home_struct))
            
        state_list.append(entry('peak_state', repeat=self.N))
        
        w = struct_symMX(state_list)

        return w
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)
        
        for home in self.homes:
            w0[home,'room_temp',:] = self.params[home]['initial_state']['room_temp']
            w0[home,'wall_temp',:] = self.params[home]['initial_state']['wall_temp']
            
        w0['peak_state',:] = self.params['peak']['initial_state']['peak_state'] #self.params['initial_state']['peak_state']
        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        
        for home in self.homes:
            lbw[home, 'P_hp', :] = 0
            ubw[home, 'P_hp', :] = self.params[home]['bounds']['P_max']
            
            lbw[home, 'slack_temp', :] = 0
        
        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            p_num[home,'energy_weight'] = opt_params['energy_weight']
            p_num[home,'comfort_weight'] = opt_params['comfort_weight']
            p_num[home,'slack_min_weight'] = opt_params['slack_min_weight']
            p_num[home,'rho_out'] = opt_params['rho_out']
            p_num[home,'rho_in'] = opt_params['rho_in']
            p_num[home,'COP'] = opt_params['COP']
        
        p_num['peak_state', 'peak_weight'] = \
            self.params['peak']['opt_params']['peak_weight']
        
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        for home in self.homes:
            traj_full[home] = {}
            traj_full[home]['room_temp'] = []
            traj_full[home]['wall_temp'] = []
            traj_full[home]['P_hp'] = []
        traj_full['peak_state'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
                
        for home in self.homes:
            for k in range(self.N-1):
                J += self.p[home, 'comfort_weight'] * \
    (self.w[home, 'room_temp', k+1] - self.p[home, 'reference_temperature', k+1])**2
                J += self.p[home,'slack_min_weight']*self.w[home,'slack_temp',k+1]**2
    
            # for k in range(self.N-1): 
                J += self.p[home, 'energy_weight'] * \
                    self.p[home, 'spot_price', k] * self.w[home, 'P_hp', k]

        # for k in range(self.N-1): # Old, peak state N-1 length
        #     J += self.p['peak_state', 'peak_weight'] * self.w['peak_state', k]
        for k in range(self.N-1):
            J += self.p['peak_state', 'peak_weight'] * self.w['peak_state', k+1]
            
        
        return J
    
    def get_constraint_functions(self):

        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for home in self.homes:
            for k in range(self.N - 1):
                rho_out = self.p[home, 'rho_out']
                rho_in = self.p[home, 'rho_in']
                wall = self.w[home, 'wall_temp', k]
                room = self.w[home, 'room_temp', k]
                Pow = self.w[home, 'P_hp', k]
                out_temp = self.p[home, 'outdoor_temperature', k]
                COP = self.p[home, 'COP']
                
                wall_plus = self.wall_func(
                    rho_out, 
                    rho_in, 
                    wall, 
                    room, 
                    out_temp
                    )
                room_plus = self.room_func(
                    rho_in,
                    room,
                    wall,
                    COP,
                    Pow
                    )
                
                g.append(wall_plus - self.w[home, 'wall_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                g.append(room_plus - self.w[home, 'room_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                
                g.append(self.p[home, 'min_temperature', k] 
                        - self.w[home, 'room_temp', k] 
                        - self.w[home, 'slack_temp', k])
                lbg.append(-inf)
                ubg.append(0)
            
        for k in range(self.N - 1):
            power_sum = 0
            for home in self.homes:
                power_sum += self.w[home, 'P_hp', k]
                power_sum += self.p[home, 'ext_power', k]
            g.append(self.w['peak_state', k+1] - power_sum) # peak state must be greater than sum of power at k
            lbg.append(0)
            ubg.append(inf)
        # for k in range(self.N - 2): # Old, shorter peak length
        #     g.append(self.w['peak_state', k+1] - self.w['peak_state', k])
        #     lbg.append(0)
        #     ubg.append(inf)
        for k in range(self.N - 1): # Flipped ineq
            g.append(self.w['peak_state', k] - self.w['peak_state', k+1])
            lbg.append(-inf)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def update_trajectory(self):
        for home in self.homes:
            self.traj_full[home]['room_temp'].append(self.w_opt[home,'room_temp',0].__float__())
            self.traj_full[home]['wall_temp'].append(self.w_opt[home,'wall_temp',0].__float__())
            self.traj_full[home]['P_hp'].append(self.w_opt[home,'P_hp',0].__float__())
        self.traj_full['peak_state'].append(self.w_opt['peak_state',0].__float__())

    def update_initial_state(self):
        for home in self.homes:
            self.w0[home,'room_temp',:self.N-1]=self.w_opt[home,'room_temp',1:]
            self.w0[home,'room_temp',-1]=self.w_opt[home,'room_temp', -1]
            
            self.w0[home,'wall_temp',:self.N-1]=self.w_opt[home,'wall_temp',1:]
            self.w0[home,'wall_temp',-1]=self.w_opt[home,'wall_temp', -1]
            
            self.w0[home,'P_hp',:self.N-2]=self.w_opt[home,'P_hp',1:]
            self.w0[home,'P_hp',-1]=self.w_opt[home,'P_hp', -1]
            
        # self.w0['peak_state', :self.N-2] = self.w_opt['peak_state', 1:] # Old peak length
        # self.w0['peak_state', -1] = self.w_opt['peak_state', -1]
        self.w0['peak_state', :self.N-1] = self.w_opt['peak_state', 1:]
        self.w0['peak_state', -1] = self.w_opt['peak_state', -1]
        
    def update_constraints(self):
        for home in self.homes:
            self.lbw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            self.ubw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            
            self.lbw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]
            self.ubw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]
            
        self.lbw['peak_state', 0] = self.w0['peak_state', 0] # New constraints
        self.ubw['peak_state', 0] = self.w0['peak_state', 0]
            
        # self.lbw['peak_state', 0] = round(float(self.w_opt['peak_state',0]), 6)
            
    def update_parameters(self, t):
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            self.p_num[home, 'outdoor_temperature'] = (
                opt_params['outdoor_temperature'][t:t+self.N]
            )
            self.p_num[home, 'reference_temperature'] = (
                opt_params['reference_temperature'][t:t+self.N]
            )
            self.p_num[home, 'min_temperature'] = (
            opt_params['min_temperature'][t:t+self.N]
            )
            self.p_num[home, 'spot_price'] = (
                opt_params['spot_price'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power'] = (
                opt_params['ext_power_avg'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power', 0] = (
                opt_params['ext_power_real'][t]
            )
        return

    def solve_optimization(self):
        """Completes optimization for one MPC step

        Returns:
            x (list): optimized state variables
            f (list): optimal cost function value
        """
        solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.p_num
            )
        # print(f'home action PID: {os.getpid()}')
        assert self.solver.stats()['success']
        lam_g = np.array(solution['lam_g']).flatten()
        lam_pk = lam_g[-99:]
        return solution['x'], solution['f']
  
        
class MPCCentralizedHomePeakQuadratic(MPCCentralizedHomePeak):
    def get_cost_function(self):
        J = 0     
        for home in self.homes: # Changed to accomodate state cost from k=1 and outwards
            for k in range(self.N-1):
                J += self.p[home, 'comfort_weight'] * \
    (self.w[home, 'room_temp', k+1] - self.p[home, 'reference_temperature', k+1])**2
                J += self.p[home,'slack_min_weight']*self.w[home,'slack_temp',k+1]**2
                
                # J += self.p['peak_state', 'peak_weight'] * self.w['peak_state', k+1]**2 Should not be per home
            
                J += self.p[home, 'energy_weight'] * \
                    self.p[home, 'spot_price', k] * self.w[home, 'P_hp', k]

        for k in range(self.N-1): 
            J += self.p['peak_state', 'peak_weight'] * self.w['peak_state', k+1]**2

            
        return J


class MPCCentralizedHome(MPC):
    
    def __init__(self, N: int, T: int, name: str, params: dict):
        homes = list(params.keys())
        homes = list(params.keys())
        if 'peak' in homes: homes.remove('peak')
        if 'max_total_power' in homes: homes.remove('max_total_power')
        self.homes = homes
        self.homes = homes
        super().__init__(N, T, name, params)
        
    def get_parameters_structure(self):
        param_list = []
        param_list.append(entry('max_total_power'))
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1),
            ])
            param_list.append(entry(home, struct=home_struct))
            
        
        p = struct_symMX(param_list)
        
        return p
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall_temp')
        room = MX.sym('room_temp')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function(
            'wall_plus', 
            [rho_out, rho_in, wall, room, OutTemp],
            [wall_plus]
            )

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function(
            'room_plus',
            [rho_in, room, wall, COP, Pow],
            [room_plus]
            )

        return wall_func, room_func

    def get_decision_variables(self):
        state_list = []
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('room_temp', repeat=self.N),
                entry('wall_temp', repeat=self.N),
                entry('slack_temp', repeat=self.N),
                entry('P_hp', repeat=self.N-1)
            ])
            state_list.append(entry(home, struct=home_struct))
        
        w = struct_symMX(state_list)
        return w
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)
        
        for home in self.homes:
            w0[home,'room_temp',:] = self.params[home]['initial_state']['room_temp']
            w0[home,'wall_temp',:] = self.params[home]['initial_state']['wall_temp']
            
        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        
        for home in self.homes:
            lbw[home, 'P_hp', :] = 0
            ubw[home, 'P_hp', :] = self.params[home]['bounds']['P_max']
            
            lbw[home, 'slack_temp', :] = 0
        
        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            p_num[home,'energy_weight'] = opt_params['energy_weight']
            p_num[home,'comfort_weight'] = opt_params['comfort_weight']
            p_num[home,'slack_min_weight'] = opt_params['slack_min_weight']
            p_num[home,'rho_out'] = opt_params['rho_out']
            p_num[home,'rho_in'] = opt_params['rho_in']
            p_num[home,'COP'] = opt_params['COP']
            
        p_num['max_total_power'] = self.params['max_total_power']
   
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        for home in self.homes:
            traj_full[home] = {}
            traj_full[home]['room_temp'] = []
            traj_full[home]['wall_temp'] = []
            traj_full[home]['P_hp'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
                
        for home in self.homes:
            for k in range(self.N-1):
                J += self.p[home, 'comfort_weight'] * \
    (self.w[home, 'room_temp', k+1] - self.p[home, 'reference_temperature', k+1])**2
    
                J += self.p[home,'slack_min_weight']*self.w[home,'slack_temp',k+1]**2

                J += self.p[home, 'energy_weight'] * \
                    self.p[home, 'spot_price', k] * self.w[home, 'P_hp', k]

        return J
    
    def get_constraint_functions(self):

        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for home in self.homes:
            for k in range(self.N - 1):
                rho_out = self.p[home, 'rho_out']
                rho_in = self.p[home, 'rho_in']
                wall = self.w[home, 'wall_temp', k]
                room = self.w[home, 'room_temp', k]
                Pow = self.w[home, 'P_hp', k]
                out_temp = self.p[home, 'outdoor_temperature', k]
                COP = self.p[home, 'COP']
                
                wall_plus = self.wall_func(
                    rho_out, 
                    rho_in, 
                    wall, 
                    room, 
                    out_temp
                    )
                room_plus = self.room_func(
                    rho_in,
                    room,
                    wall,
                    COP,
                    Pow
                    )
                
                g.append(wall_plus - self.w[home, 'wall_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                g.append(room_plus - self.w[home, 'room_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                
                g.append(self.p[home, 'min_temperature', k] 
                        - self.w[home, 'room_temp', k] 
                        - self.w[home, 'slack_temp', k])
                lbg.append(-inf)
                ubg.append(0)
            
        for k in range(self.N - 1):
            power_sum = 0
            for home in self.homes:
                power_sum += self.w[home, 'P_hp', k]
                power_sum += self.p[home, 'ext_power', k]
            g.append(power_sum - self.p['max_total_power']) # Constraint that is dualized
            lbg.append(-inf)
            ubg.append(0)

        return g, lbg, ubg
    
    def update_trajectory(self):
        for home in self.homes:
            self.traj_full[home]['room_temp'].append(self.w_opt[home,'room_temp',0].__float__())
            self.traj_full[home]['wall_temp'].append(self.w_opt[home,'wall_temp',0].__float__())
            self.traj_full[home]['P_hp'].append(self.w_opt[home,'P_hp',0].__float__())

    def update_initial_state(self):
        for home in self.homes:
            self.w0[home,'room_temp',:self.N-1]=self.w_opt[home,'room_temp',1:]
            self.w0[home,'room_temp',-1]=self.w_opt[home,'room_temp', -1]
            
            self.w0[home,'wall_temp',:self.N-1]=self.w_opt[home,'wall_temp',1:]
            self.w0[home,'wall_temp',-1]=self.w_opt[home,'wall_temp', -1]
            
            self.w0[home,'P_hp',:self.N-2]=self.w_opt[home,'P_hp',1:]
            self.w0[home,'P_hp',-1]=self.w_opt[home,'P_hp', -1]
        
    def update_constraints(self):
        for home in self.homes:
            self.lbw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            self.ubw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            
            self.lbw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]
            self.ubw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]
            
    def update_parameters(self, t):
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            self.p_num[home, 'outdoor_temperature'] = (
                opt_params['outdoor_temperature'][t:t+self.N]
            )
            self.p_num[home, 'reference_temperature'] = (
                opt_params['reference_temperature'][t:t+self.N]
            )
            self.p_num[home, 'min_temperature'] = (
            opt_params['min_temperature'][t:t+self.N]
            )
            self.p_num[home, 'spot_price'] = (
                opt_params['spot_price'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power'] = (
                opt_params['ext_power_avg'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power', 0] = (
                opt_params['ext_power_real'][t]
            )
        return

    def solve_optimization(self):
        solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.p_num
            )
        # print(f'home action PID: {os.getpid()}')
        assert self.solver.stats()['success']
        lam_g = np.array(solution['lam_g']).flatten()
        lam_pk = lam_g[-self.N+1:]
        return solution['x'], solution['f']


class MPCCentralizedHomeSinglePeak(MPC):
    
    def __init__(self, N: int, T: int, name: str, params: dict):
        homes = list(params.keys())
        if 'peak' in homes: homes.remove('peak')
        if 'max_total_power' in homes: homes.remove('max_total_power')
        self.homes = homes
        super().__init__(N, T, name, params)
        
    def get_parameters_structure(self):
        param_list = []
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('slack_min_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('min_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('ext_power', repeat=self.N-1)
            ])
            param_list.append(entry(home, struct=home_struct))
            
        peak_struct = struct_symMX([
            entry('peak_weight')
            ])
        param_list.append(entry('peak_state', struct=peak_struct))
        
        p = struct_symMX(param_list)
        
        return p
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall_temp')
        room = MX.sym('room_temp')
        OutTemp = MX.sym('OutTemp')

        rho_out = MX.sym('rho_out')
        rho_in = MX.sym('rho_in')
        wall_plus = wall + rho_out * (OutTemp - wall) + rho_in * (room - wall)
        wall_func = Function(
            'wall_plus', 
            [rho_out, rho_in, wall, room, OutTemp],
            [wall_plus]
            )

        COP = MX.sym('COP')
        Pow = MX.sym('Pow')
        room_plus = room + rho_in * (wall - room) + COP * Pow
        room_func = Function(
            'room_plus',
            [rho_in, room, wall, COP, Pow],
            [room_plus]
            )

        return wall_func, room_func

    def get_decision_variables(self):
        state_list = []
        
        for home in self.homes:
            home_struct = struct_symMX([
                entry('room_temp', repeat=self.N),
                entry('wall_temp', repeat=self.N),
                entry('slack_temp', repeat=self.N),
                entry('P_hp', repeat=self.N-1)
            ])
            state_list.append(entry(home, struct=home_struct))
            
        state_list.append(entry('peak_state'))
        
        w = struct_symMX(state_list)

        return w
    
    def get_initial_state(self):
        w0 = copy(self.w)(0)
        
        for home in self.homes:
            w0[home,'room_temp',:] = self.params[home]['initial_state']['room_temp']
            w0[home,'wall_temp',:] = self.params[home]['initial_state']['wall_temp']
            
        w0['peak_state'] = self.params['peak']['initial_state']['peak_state'] #self.params['initial_state']['peak_state']
        return w0
    
    def get_state_bounds(self):
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        
        for home in self.homes:
            lbw[home, 'P_hp', :] = 0
            ubw[home, 'P_hp', :] = self.params[home]['bounds']['P_max']
            
            lbw[home, 'slack_temp', :] = 0
        
        return lbw, ubw
    
    def get_numerical_parameters(self):
        p_num = self.p(0)
        
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            p_num[home,'energy_weight'] = opt_params['energy_weight']
            p_num[home,'comfort_weight'] = opt_params['comfort_weight']
            p_num[home,'slack_min_weight'] = opt_params['slack_min_weight']
            p_num[home,'rho_out'] = opt_params['rho_out']
            p_num[home,'rho_in'] = opt_params['rho_in']
            p_num[home,'COP'] = opt_params['COP']
        
        p_num['peak_state', 'peak_weight'] = \
            self.params['peak']['opt_params']['peak_weight']
        
        return p_num
    
    def get_trajectory_structure(self):
        traj_full = {}
        for home in self.homes:
            traj_full[home] = {}
            traj_full[home]['room_temp'] = []
            traj_full[home]['wall_temp'] = []
            traj_full[home]['P_hp'] = []
        traj_full['peak_state'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
                
        for home in self.homes:
            for k in range(self.N-1):
                J += self.p[home, 'comfort_weight'] * \
    (self.w[home, 'room_temp', k+1] - self.p[home, 'reference_temperature', k+1])**2
    
                J += self.p[home,'slack_min_weight']*self.w[home,'slack_temp',k+1]**2

                J += self.p[home, 'energy_weight'] * \
                    self.p[home, 'spot_price', k] * self.w[home, 'P_hp', k]

        J += self.p['peak_state', 'peak_weight'] * self.w['peak_state']
        
        return J
    
    def get_constraint_functions(self):

        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for home in self.homes:
            for k in range(self.N - 1):
                rho_out = self.p[home, 'rho_out']
                rho_in = self.p[home, 'rho_in']
                wall = self.w[home, 'wall_temp', k]
                room = self.w[home, 'room_temp', k]
                Pow = self.w[home, 'P_hp', k]
                out_temp = self.p[home, 'outdoor_temperature', k]
                COP = self.p[home, 'COP']
                
                wall_plus = self.wall_func(
                    rho_out, 
                    rho_in, 
                    wall, 
                    room, 
                    out_temp
                    )
                room_plus = self.room_func(
                    rho_in,
                    room,
                    wall,
                    COP,
                    Pow
                    )
                
                g.append(wall_plus - self.w[home, 'wall_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                g.append(room_plus - self.w[home, 'room_temp', k+1])
                lbg.append(0)
                ubg.append(0)
                
                g.append(self.p[home, 'min_temperature', k] 
                        - self.w[home, 'room_temp', k] 
                        - self.w[home, 'slack_temp', k])
                lbg.append(-inf)
                ubg.append(0)
            
        for k in range(self.N - 1):
            power_sum = 0
            for home in self.homes:
                power_sum += self.w[home, 'P_hp', k]
                power_sum += self.p[home, 'ext_power', k]
            g.append(self.w['peak_state'] - power_sum) # peak state must be greater than sum of power at k
            lbg.append(0)
            ubg.append(inf)
                
        return g, lbg, ubg
    
    def update_trajectory(self):
        for home in self.homes:
            self.traj_full[home]['room_temp'].append(self.w_opt[home,'room_temp',0].__float__())
            self.traj_full[home]['wall_temp'].append(self.w_opt[home,'wall_temp',0].__float__())
            self.traj_full[home]['P_hp'].append(self.w_opt[home,'P_hp',0].__float__())
        self.traj_full['peak_state'].append(self.w_opt['peak_state'].__float__())

    def update_initial_state(self):
        for home in self.homes:
            self.w0[home,'room_temp',:self.N-1]=self.w_opt[home,'room_temp',1:]
            self.w0[home,'room_temp',-1]=self.w_opt[home,'room_temp', -1]
            
            self.w0[home,'wall_temp',:self.N-1]=self.w_opt[home,'wall_temp',1:]
            self.w0[home,'wall_temp',-1]=self.w_opt[home,'wall_temp', -1]
            
            self.w0[home,'P_hp',:self.N-2]=self.w_opt[home,'P_hp',1:]
            self.w0[home,'P_hp',-1]=self.w_opt[home,'P_hp', -1]
            
        # self.w0['peak_state', :self.N-1] = self.w_opt['peak_state', 1:]
        # self.w0['peak_state', -1] = self.w_opt['peak_state', -1]
        
    def update_constraints(self):
        for home in self.homes:
            self.lbw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            self.ubw[home, 'wall_temp', 0] = self.w0[home, 'wall_temp', 0]
            
            self.lbw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]
            self.ubw[home, 'room_temp', 0] = self.w0[home, 'room_temp', 0]

            
    def update_parameters(self, t):
        for home in self.homes:
            opt_params = self.params[home]['opt_params']
            self.p_num[home, 'outdoor_temperature'] = (
                opt_params['outdoor_temperature'][t:t+self.N]
            )
            self.p_num[home, 'reference_temperature'] = (
                opt_params['reference_temperature'][t:t+self.N]
            )
            self.p_num[home, 'min_temperature'] = (
            opt_params['min_temperature'][t:t+self.N]
            )
            self.p_num[home, 'spot_price'] = (
                opt_params['spot_price'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power'] = (
                opt_params['ext_power_avg'][t:t+self.N-1]
            )
            self.p_num[home, 'ext_power', 0] = (
                opt_params['ext_power_real'][t]
            )
        return

    def solve_optimization(self):
        """Completes optimization for one MPC step

        Returns:
            x (list): optimized state variables
            f (list): optimal cost function value
        """
        solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.p_num
            )
        # print(f'home action PID: {os.getpid()}')
        assert self.solver.stats()['success']
        lam_g = np.array(solution['lam_g']).flatten()
        lam_pk = lam_g[-self.N+1:]
        return solution['x'], solution['f']


class ProximalGradientSolver():
    """Proximal Gradient solver for projecting  varaible onto a feasible
    plane
    
    """
    def __init__(self, N: int, peak_weight: float):
        """Create an MPC object

        Args:
            N (int): optimization window length
            peak_weight (float): weight of lone peak state in org. opt. prob.
        """
        self.N = N
        self.peak_weight = peak_weight
        
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.w0 = self.get_initial_state()
        self.lbw, self.ubw = self.get_state_bounds()
        self.p_num = self.get_numerical_parameters()
        self.w_opt = copy(self.w)(0)
        
        self.solver = self.get_solver()
        
        dumps(self) # Another quick fix for serializability
        
    def get_decision_variables(self):
        """Builds the decision variable
        
        Returns:
            w (struct_symMX): decision variable
        """
        w = struct_symMX([
        entry('mu_proj', repeat=self.N-1)
        ])
        return w
    
    def get_parameters_structure(self):
        """Builds the the parameters structure
        
        Returns: 
            p (struct_symMX): parameter structure
        """
        p = struct_symMX([
        entry('mu_plus', repeat=self.N-1),
        entry('peak_weight')
        ])
        return p
    
    def get_initial_state(self):
        """Builds initial decision variable state from params
        
        Returns:
            w0 (DMStruct): initial state for optimization
        """
        w0 = self.w(0)
        return w0
    
    def get_state_bounds(self):
        """Builds bounds on the state w
        
        Returns:
            lbw (DMStruct): lower bounds on w
            ubw (DMStruct): upper bounds on w
        """
        lbw = copy(self.w)(-inf)
        ubw = copy(self.w)(inf)
        lbw['mu_proj', :] = 0 # Projection to keep mu positive
        return lbw, ubw
    
    def get_numerical_parameters(self):
        """Builds numerical optimization parameters from params
        
        Returns:
            p_num (DMStruct): numerical parameters for optimization
        """
        p_num = self.p(0)
        p_num['peak_weight'] = self.peak_weight
        return p_num
    
    def get_cost_funtion(self):
        """Builds the cost function
        
        Returns:
            J (MX): cost function
        """
        J = 0
        for k in range(self.N-1):
            J += 0.5 * (self.w['mu_proj', k] - self.p['mu_plus', k])**2
        return J
    
    def get_constraint_functions(self):
        """Builds constraint functions and bounds
        
        Returns:
            g (list(MX)): constraint functions
            lbg (list): lower bounds on g
            ubg (list): upper bounds on g
        """
        g = []
        lbg = []
        ubg = []
        
        mu_sum = 0
        for k in range(self.N - 1):    
            mu_sum += self.p['mu_plus', k]
        # 0 \leq w - 1^T \mu \leq 0
        g.append(self.p['weight'] - mu_sum) # Projection to meet function
        lbg.append(0)
        ubg.append(0)
                
        return g, lbg, ubg
    
    def get_solver(self):
        """Builds casadi non linear programming solver
        
        Returns:
            solver (casadi::Function): NLP solver for opt. problem
        """
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_function(), 
            'x': self.w, 
            'g': vertcat(*(g)), 
            'p': self.p
            }
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def solve_optimization(self):
        """Completes optimization for one MPC step

        Returns:
            x (list): optimized state variables
            f (list): optimal cost function value
        """
        solution = self.solver(
            x0=self.w0, 
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.p_num
            )
        assert self.solver.stats()['success']
        # print(f'home action PID: {os.getpid()}')
        return solution['x']#, solution['f']
    
    def set_optimal_state(self, w_res):
        """Set internal optimal state with result from optimization
        
        """
        if not isinstance(w_res, structure3.DMStruct):
            w_res = self.w(w_res)
        self.w_opt = w_res
        
    def set_initial_state(self, w_res):
        """Set internal initial state
        
        """
        if not isinstance(w_res, structure3.DMStruct):
            w_res = self.w(w_res)
        self.w0 = w_res
    

    def update_parameters_generic(self, **kwargs):
        """Modulary updates parameters, assuming they are already of the right
        size.
        
        """
        for key, value in kwargs.items():
            self.p_num[key] = list(value) 