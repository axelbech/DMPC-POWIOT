from casadi import *
from casadi.tools import *

class Peak_state_solver():
    def __init__(self, n_steps, n_homes, P_max):
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
    
    
if __name__=='__main__':
    N = 20
    M = 5
    p_max = 1.5
    lbw = []
    ubw = []
    for k in range(N):
        lbw.append(0)
        ubw.append(M * p_max)
    peak_solver = Peak_state_solver(n_steps=N, n_homes=M, P_max=p_max)
    w0 = peak_solver.w(1)
    p_num = peak_solver.p(0)
    p_num['weight'] = 1
    p_num['dual_variable', :] = 10
    x = peak_solver.solve_peak_problem(w0,lbw,ubw,p_num)
    print(x)