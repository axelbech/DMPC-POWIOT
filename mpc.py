from weakref import ref
from casadi import *
from casadi.tools import *
import numpy as np

class MPC():
    """Model predictive control template for model predictive control using 
    CasADi symbolics and IPOPT for optimization
    
    """
    def __init__(self, N, name):
        """Create an MPC object

        Args:
            N (int): optimization window length
            name (str): instance name for parameter indexing
        """
        self.N = N
        self.name = name

        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.w0 = self.w(0)
        self.lbw = self.w(-inf)
        self.ubw = self.w(inf)
        self.w_opt = self.w(0)
        self.p_num = self.p(0)
        
        self.solver = self.get_solver()
        
        self.traj_full = self.get_trajectory_structure()
        
    def get_parameters_structure(self):
        """Builds the the parameters structure
        
        Returns: 
            p (struct_symMX): parameter structure
        """
        pass

    def get_decision_variables(self):
        """Builds the decision variable
        
        Returns:
            w (struct_symMX): decision variable
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
    
    def get_MPC_action(self):
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
        return solution['x'], solution['f']
    
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
    
    def update_optimal_state(self, w_opt):
        """Update internal optimal state with result from optimization
        
        """
        if not isinstance(w_opt, structure3.DMStruct):
            w_opt = self.w(w_opt)
        self.w_opt = w_opt
        
    def update_trajectory(self):
        """Update trajectory with w_opt"""
        pass
    
    def update_parameters_generic(self, **kwargs):
        """Modulary updates parameters, assuming they are already of the right
        size.
        
        """
        for key, value in kwargs.items():
            self.p_num[key] = list(value) 
            
    def update_parameters(self, p_ext, t):
        """Update time variant internal parameters from external source

        Args:
            p_ext (dict): external parameters structure
            t (int): time step
        """
        pass
    
    def set_parameters(self, p_ext):
        """Set time invariant internal parameters from external source

        Args:
            p_ext (dict): external parameters structure
        """
        pass
        

class MPCSingleHome(MPC):
        
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1)
            ])
        
    def get_dynamics_functions(self):
        wall = MX.sym('wall')
        room = MX.sym('room')
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
        MPC_states = struct_symMX([entry('room'), entry('wall')])
        MPC_inputs = struct_symMX([entry('P_hp')])

        w = struct_symMX([
        entry('state', struct=MPC_states, repeat=self.N),
        entry('input', struct=MPC_inputs, repeat=self.N - 1)
        ])
        return w
    
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['room'] = []
        traj_full['wall'] = []
        traj_full['P_hp'] = []
        return traj_full
    
    def get_cost_function(self):
        J = 0
        for k in range(self.N): 
            J += self.p['comfort_weight'] * \
            (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['energy_weight'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
        
        return J
    
    def get_constraint_functions(self):
        rho_out = self.p['rho_out']
        rho_in = self.p['rho_in']
        g = []
        lbg = []
        ubg = []
        self.wall_func, self.room_func = self.get_dynamics_functions()
        
        for k in range(self.N - 1):
            wall = self.w['state', k, 'wall']
            room = self.w['state', k, 'room']
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
            
            g.append(wall_plus - self.w['state', k+1, 'wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def update_trajectory(self):
        self.traj_full['room'].append(self.w_opt['state',0,'room'])
        self.traj_full['wall'].append(self.w_opt['state',0,'wall'])
        self.traj_full['P_hp'].append(self.w_opt['state',0,'room'])
    

    def update_initial_state(self):
        self.w0['state', :N-1] = self.w_opt['state', 1:]
        self.w0['state', -1] = self.w_opt['state', -1]
        
        self.w0['input', :N-2] = self.w_opt['input', 1:]
        self.w0['input', -1] = self.w_opt['input', -1]
        
    def update_constraints(self):
        self.lbw['state', 0, 'wall'] = self.w0['state', 0, 'wall']
        self.ubw['state', 0, 'wall'] = self.w0['state', 0, 'wall']
        
        self.lbw['state', 0, 'room'] = self.w0['state', 0, 'room']
        self.ubw['state', 0, 'room'] = self.w0['state', 0, 'room']
        
    def update_parameters(self, p_ext, t):
        specific = p_ext['controller_specific'][self.name]
        self.p_num['outdoor_temperature'] = (
            p_ext['outdoor_temperature'][t:t+self.N]
        )
        self.p_num['reference_temperature'] = (
            specific['reference_temperature'][t:t+self.N]
        )
        self.p_num['spot_price'] = (
            p_ext['spot_price'][t:t+self.N-1]
        )
    
    def set_parameters(self, p_ext):
        specific = p_ext['controller_specific'][self.name]
        self.p_num['energy_weight'] = specific['energy_weight']
        self.p_num['comfort_weight'] = specific['comfort_weight']
        self.p_num['rho_out'] = specific['rho_out']
        self.p_num['rho_in'] = specific['rho_in']
        self.p_num['COP'] = specific['COP']

            
class MPCSingleHomeDistributed(MPCSingleHome):
        
    def get_parameters_structure(self):
        return struct_symMX([
                entry('energy_weight'),
                entry('comfort_weight'),
                entry('rho_out'),
                entry('rho_in'),
                entry('COP'),
                entry('outdoor_temperature', repeat=self.N),
                entry('reference_temperature', repeat=self.N),
                entry('spot_price', repeat=self.N-1),
                entry('dual_variable', repeat=self.N-1)
            ])
    
    def get_cost_function(self):
        J = 0
        for k in range(self.N): 
            J += self.p['comfort_weight'] * \
            (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['energy_weight'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
            J += self.p['dual_variable', k] * self.w['input', k, 'P_hp'] # From dual decomposition, dual_var like a power price
        
        return J
    
    def dummy_func(self):
        print('starting func')
        for i in range(10000):
            self.w0['state', 0, 'room'] = 19
            self.w0['state', 0, 'room'] = 20
        print('finishing func')
        return self.w0


class MPCPeakStateDistributed(MPC):
        
    def get_decision_variables(self):
        return struct_symMX([entry('peak', repeat=self.N-1)])
    
    def get_parameters_structure(self):
        return struct_symMX([
            entry('peak_weight'), 
            entry('dual_variable', repeat=self.N-1)
            ])

    def get_cost_function(self):
        J = 0
        for k in range(self.N-1):
            J -= self.p['dual_variable', k] * self.w['peak', k]
            J += self.p['peak_weight'] * self.w['peak', k]
        return J
            
    def get_constraint_functions(self):
        g = []
        lbg = []
        ubg = []
        for k in range(self.N - 2):
            g.append(self.w['peak', k+1] - self.w['peak', k])
            lbg.append(0)
            ubg.append(inf)
        return g, lbg, ubg
    
    def get_trajectory_structure(self):
        traj_full = {}
        traj_full['peak'] = []
        return traj_full
    
    def get_dual_update_contribution(self):
        return -np.array(vertcat(*self.w_opt['peak',:])).flatten()
    
    def update_trajectory(self):
        self.traj_full['peak'].append(self.w_opt['peak',0])
    
    def update_initial_state(self):
        self.w0['peak', :N-2] = self.w_opt['peak', 1:]
        self.w0['peak', -1] = self.w_opt['peak', -1]
        
    def update_constraints(self):
        self.lbw['peak', 0] = self.w0['peak', 0]
        self.ubw['peak', 0] = self.w0['peak', 0]
    
    def set_parameters(self, p_ext):
        specific = p_ext['controller_specific'][self.name]
        self.p_num['peak_weight'] = specific['peak_weight']

        

if __name__ == '__main__':
    from time import time, sleep
    from pickle import loads, dumps
    from concurrent.futures import ProcessPoolExecutor
    N = 10
    reference_temperature = [23 for _ in range(N)]
    outdoor_temperature = [10 for _ in range(N)]
    spot_price = [-(x-N/2)**2 + 50 for x in range(N-1)]
    dual_variable = spot_price
    
    # a = MPC(N=N,name='gen')
    b = MPCSingleHome(N=N,name='single')
    c = MPCSingleHomeDistributed(N=N,name='singleDistributed')
    d = MPCPeakStateDistributed(N=N,name='peak')