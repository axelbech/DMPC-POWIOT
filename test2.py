from casadi import *
from casadi.tools import *

from pickle import loads, dumps
from concurrent.futures import ProcessPoolExecutor

class myClass():
    def __init__(self):
        self.s = struct_symMX([entry('x')])
        # self.s_num = self.s(0)
        self.solver = self.set_solver()
        
    def set_solver(self):
        J = 0
        J += self.s['x']
        mpc_problem = {'f': J, 'x': self.s}
        opts = {'ipopt.print_level':0, 'print_time':0}
        return nlpsol('solver', 'ipopt', mpc_problem, opts)
    
    def get_action(self, w0, lbw, ubw):
        return self.solver(x0=w0, lbx=lbw, ubx=ubw)
    
    def execute_actions(self):
        s0 = c.s(3)
        lbw = 0
        ubw = 5
        with ProcessPoolExecutor(max_workers=8) as executor:
            future = executor.submit(self.solver, x0=s0, lbx=lbw, ubx=ubw)
            print(future.result())



# s0 = c.s(3)
# lbw = 0
# ubw = 5

if __name__ == '__main__':
    c = myClass()
    c.execute_actions()
    # print(loads(dumps(s0)))
    # with ProcessPoolExecutor() as executor:
    #     future = executor.submit(c.get_action,s0.master,lbw,ubw)
    #     print(future.result())

# print(c.get_action(s0, lbw, ubw))

#does work
# print(loads(dumps(c.s_num)))