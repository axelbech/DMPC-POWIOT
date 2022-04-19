from weakref import ref
from casadi import *
from casadi.tools import *
from concurrent.futures import ProcessPoolExecutor

class MPC():
    def __init__(self, N):
        self.N = N

        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.w0 = self.w(0)
        self.lbw = self.w(-inf)
        self.ubw = self.w(inf)
        self.w_opt = self.w(0)
        self.p_num = self.p(0)
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        return None

    def get_decision_variables(self):
        return None
    
    def get_cost_funtion(self):
        return None
    
    def get_constraint_functions(self):
        return None
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_funtion(), 
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
    
    
    def update_parameters(self, **kwargs):
        """Modulary updates parameters, assuming they are already of the right
        size.
        
        """
        for key, value in kwargs.items():
            self.p_num[key] = list(value) 
            
            
class MPCSingleHomeDistributedInherited(MPC):
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([
            entry('energy'), 
            entry('comfort')
            ])
        p_model = struct_symMX([
            entry('rho_out'), 
            entry('rho_in'),
            entry('COP')])
        
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
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N): 
            J += self.p['weights', 'comfort'] * \
            (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['weights', 'energy'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
            J += self.p['dual_variable', k] * self.w['input', k, 'P_hp'] # From dual decomposition, dual_var like a power price
        
        return J
    
    def get_constraint_functions(self):
        rho_out = self.p['model', 'rho_out']
        rho_in = self.p['model', 'rho_in']
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
                self.p['model', 'COP'],
                Pow
                )
            
            g.append(wall_plus - self.w['state', k+1, 'wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    
    def dummy_func(self):
        print('starting func')
        for i in range(10000):
            self.w0['state', 0, 'room'] = 19
            self.w0['state', 0, 'room'] = 20
        print('finishing func')
        return self.w0
    
    
    @staticmethod
    def update_initial_state(w0, w, N):
        """Using the result state from an optimization w at time t, update the
        start state w0 for time t+1
        
        """
        w0['state', :N-1] = w['state', 1:]
        w0['state', -1] = w['state', -1]
        
        w0['input', :N-2] = w['input', 1:]
        w0['input', -1] = w['input', -1]
        
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        """The first state should not change
        
        """
        lbw['state', 0, 'wall'] = w0['state', 0, 'wall']
        ubw['state', 0, 'wall'] = w0['state', 0, 'wall']
        
        lbw['state', 0, 'room'] = w0['state', 0, 'room']
        ubw['state', 0, 'room'] = w0['state', 0, 'room']
        
        lbw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        ubw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        
    @classmethod   
    def prepare_MPC_action(cls, w0, w, N, lbw, ubw):
        cls.update_initial_state(w0, w, N)
        cls.update_constraints(w0, lbw, ubw)


class MPCSingleHome():
    def __init__(self, N, P_max = 1.5):
        self.N = N
        self.P_max = P_max

        self.wall_func, self.room_func = self.get_dynamics_functions()
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([
            entry('energy'), 
            entry('comfort')
            ])
        p_model = struct_symMX([
            entry('rho_out'), 
            entry('rho_in'),
            entry('COP')])
        
        return struct_symMX([
                entry('weights', struct=p_weights),
                entry('model', struct=p_model),
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
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N): 
            J += self.p['weights', 'comfort'] * \
            (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['weights', 'energy'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
        
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
                self.p['model', 'COP'],
                Pow
                )
            
            g.append(wall_plus - self.w['state', k+1, 'wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_funtion(), 
            'x': self.w, 
            'g': vertcat(*(g)), 
            'p': self.p
            }
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def get_MPC_action(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        print(f'home action PID: {os.getpid()}')
        return solution['x']
    
    @staticmethod
    def update_initial_state(w0, w, N):
        """Using the result state from an optimization w at time t, update the
        start state w0 for time t+1
        
        """
        w0['state', :N-1] = w['state', 1:]
        w0['state', -1] = w['state', -1]
        
        w0['input', :N-2] = w['input', 1:]
        w0['input', -1] = w['input', -1]
        
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        """The first state should not change, enforced with constraints"""
        lbw['state', 0, 'wall'] = w0['state', 0, 'wall']
        ubw['state', 0, 'wall'] = w0['state', 0, 'wall']
        
        lbw['state', 0, 'room'] = w0['state', 0, 'room']
        ubw['state', 0, 'room'] = w0['state', 0, 'room']
        
        lbw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        ubw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        
    @classmethod   
    def prepare_MPC_action(cls, w0, w, N, lbw, ubw):
        cls.update_initial_state(w0, w, N)
        cls.update_constraints(w0, lbw, ubw)


class MPCSingleHomeDistributed():
    def __init__(self, N, P_max = 1.5):
        self.N = N
        self.P_max = P_max

        self.wall_func, self.room_func = self.get_dynamics_functions()
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.w0 = self.w(0)
        self.w_opt = self.w(0)
        self.p_num = self.p(0)
        
        self.solver = self.get_solver()
        
    def get_parameters_structure(self):
        p_weights = struct_symMX([
            entry('energy'), 
            entry('comfort')
            ])
        p_model = struct_symMX([
            entry('rho_out'), 
            entry('rho_in'),
            entry('COP')])
        
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
    
    def get_cost_funtion(self):
        J = 0
        for k in range(self.N): 
            J += self.p['weights', 'comfort'] * \
            (self.w['state', k, 'room'] - self.p['reference_temperature',k])**2
        
        for k in range(self.N - 1): # Input not defined for the last timestep
            J += self.p['weights', 'energy'] * self.p['spot_price', k]\
                * self.w['input', k, 'P_hp']
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
                self.p['model', 'COP'],
                Pow
                )
            
            g.append(wall_plus - self.w['state', k+1, 'wall'])
            lbg.append(0)
            ubg.append(0)
            g.append(room_plus - self.w['state', k+1, 'room'])
            lbg.append(0)
            ubg.append(0)
                
        return g, lbg, ubg
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_funtion(), 
            'x': self.w, 
            'g': vertcat(*(g)), 
            'p': self.p
            }
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def get_MPC_action(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                               lbg=self.lbg, ubg=self.ubg, p=p_num)
        print(f'home action PID: {os.getpid()}')
        return solution['x']
    
    def dummy_func(self):
        print('starting func')
        for i in range(10000):
            self.w0['state', 0, 'room'] = 19
            self.w0['state', 0, 'room'] = 20
        print('finishing func')
        return self.w0
    
    def update_parameters(self, **kwargs):
        """Modulary updates parameters, assuming they are already of the right
        size.
        
        """
        for key, value in kwargs.items():
            self.p_num[key] = list(value) 
    
    
    @staticmethod
    def update_initial_state(w0, w, N):
        """Using the result state from an optimization w at time t, update the
        start state w0 for time t+1
        
        """
        w0['state', :N-1] = w['state', 1:]
        w0['state', -1] = w['state', -1]
        
        w0['input', :N-2] = w['input', 1:]
        w0['input', -1] = w['input', -1]
        
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        """The first state should not change
        
        """
        lbw['state', 0, 'wall'] = w0['state', 0, 'wall']
        ubw['state', 0, 'wall'] = w0['state', 0, 'wall']
        
        lbw['state', 0, 'room'] = w0['state', 0, 'room']
        ubw['state', 0, 'room'] = w0['state', 0, 'room']
        
        lbw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        ubw['input', 0, 'P_hp'] = w0['input', 0, 'P_hp']
        
    @classmethod   
    def prepare_MPC_action(cls, w0, w, N, lbw, ubw):
        cls.update_initial_state(w0, w, N)
        cls.update_constraints(w0, lbw, ubw)
        
        
class MPCPeakState():
    def __init__(self, N, P_max=1.5):
        self.N = N
        # self.n_homes = n_homes
        self.P_max = P_max
        
        self.w = self.get_decision_variables()
        self.p = self.get_parameters_structure()
        
        self.solver = self.get_solver()
        
    def get_decision_variables(self):
        return struct_symMX([entry('peak', repeat=self.N-1)])
    
    def get_parameters_structure(self):
        return struct_symMX([
            entry('weight'), 
            entry('dual_variable', repeat=self.N-1)
            ])

    def get_cost_function(self):
        J = 0
        # J += self.n_homes * self.n_steps * self.p['weight'] * self.w['peak', -1] # Old last state cost
        for k in range(self.N-1):
            J -= self.p['dual_variable', k] * self.w['peak', k]
            J += self.p['weight'] * self.w['peak', k] # Old last state cost
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
    
    def get_solver(self):
        g, self.lbg, self.ubg = self.get_constraint_functions()
        mpc_problem = {
            'f': self.get_cost_function(), 
            'x': self.w, 
            'g': vertcat(*(g)),
            'p': self.p
            }
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def solve_peak_problem(self, w0, lbw, ubw, p_num):
        solution = self.solver(x0=w0, lbx=lbw, ubx=ubw,
                                lbg=self.lbg, ubg=self.ubg, p=p_num)
        print(f'peak action PID: {os.getpid()}')
        return solution['x']
    
    @staticmethod
    def update_initial_state(w0, w, n_steps):
        w0['peak', :n_steps-1] = w['peak', 1:]
        w0['peak', -1] = w['peak', -1]
        
    @staticmethod
    def update_constraints(w0, lbw, ubw):
        lbw['peak', 0] = w0['peak', 0]
        ubw['peak', 0] = w0['peak', 0] 
        
    @classmethod 
    def prepare_action(cls, w0, w, n_steps, lbw, ubw):
        cls.update_initial_state(w0, w, n_steps)
        cls.update_constraints(w0, lbw, ubw)
        

if __name__ == '__main__':
    from time import time, sleep
    from pickle import loads, dumps
    from concurrent.futures import ProcessPoolExecutor
    N = 10
    c = MPCSingleHomeDistributed(N=N)
    d = MPCSingleHomeDistributed(N=N)
    reference_temperature = [23 for _ in range(N)]
    outdoor_temperature = [10 for _ in range(N)]
    spot_price = [-(x-N/2)**2 + 50 for x in range(N-1)]
    dual_variable = spot_price
    
    i = MPCSingleHomeDistributedInherited(N=N)
    i.update_parameters(spot_price=spot_price)
    g = dumps(i)
    k = loads(dumps(i))
    print(k.p_num.master)
    
    # c.update_parameters(
    #     reference_temperature=reference_temperature,
    #     outdoor_temperature=outdoor_temperature,
    #     spot_price=spot_price,
    #     dual_variable=dual_variable
    # )
    # d.update_parameters(
    #     reference_temperature=reference_temperature,
    #     outdoor_temperature=outdoor_temperature,
    #     spot_price=spot_price,
    #     dual_variable=dual_variable
    # )
    # g = dumps(c)
    # g = dumps(d)
    # with ProcessPoolExecutor() as executor:
    #     print('### First batch ###')
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
        
    #     c.w0['input', 0, 'P_hp'] = 1
        
    #     sleep(5)
    #     print('### Second batch ###')
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
    #     result_map = executor.map(MPCSingleHomeDistributed.dummy_func, [c,d], chunksize=8)
    #     # result_list = list(result_map)
    #     # w0c = result_list[0]
    #     # w0d = result_list[1]
    #     # print(w0c.master)
    #     # print(w0d.master)