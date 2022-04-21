import numpy as np

class PartitionedMPC():
    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict, 
        params: dict
        ):
        
        self.N = N
        self.T = T
        self.mpcs = mpcs
        self.params = params
    def update_mpc_constraints(self):
        for mpc in self.mpcs.values():
            mpc.update_constraints()
        
    def update_mpc_state_trajectories(self):
        for mpc in self.mpcs.values():
            mpc.update_trajectory()
            
    def update_mpc_parameters(self, t):
        for mpc in self.mpcs.values():
            mpc.update_parameters(self.params, t)
            
    def set_mpc_parameters(self):
        for mpc in self.mpcs.values():
            mpc.set_parameters(self.params)
            
            
class DistributedMPC(PartitionedMPC):

    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict, 
        params: dict,
        dual_update_constant: float
        ):
        
        super.__init__(N, T, mpcs, params)
        self.dual_variables = self.get_dual_variables()
        self.dual_variables_traj = self.get_dual_variables_trajectory()
        self.dual_update_constant = dual_update_constant
    
    def get_dual_variables(self):
        return np.zeros(self.N-1)
    
    def get_dual_variables_trajectory(self):
        return np.zeros((self.T,self.N-1))
    
    def update_dual_variables_trajectory(self, t):
        self.dual_variables_traj[t, :] = self.dual_variables
        
    def update_mpc_dual_variables(self):
        for mpc in self.mpcs.values():
            mpc.update_parameters_generic(dual_variables=self.dual_variables)
        
    def dual_decomposition(self):
        it = 0
        maxIt = 10
        
        f_tol = 1e-2 
        f_err = 1e6
        f_sum_old = 1e6
        while it < maxIt and f_err < f_tol:
            f_sum = 0
            dual_updates = np.zeros_like(self.dual_variables)
            for mpc in self.mpcs.values():
                w_opt, f_opt = mpc.solve_optimization()
                mpc.update_optimal_state(mpc.w(w_opt))
                f_sum += f_opt
                dual_updates += mpc.get_dual_update_contribution()
            dual_update_step_size = 20  / np.sqrt(1+it)
            self.dual_variables += dual_update_step_size * dual_updates
            self.update_mpc_dual_variables()
            
            f_err = np.abs(f_sum - f_sum_old)
            f_sum_old = f_sum
        
        
    def run_full(self):
        
        for t in range(self.T):
            
            self.update_mpc_parameters(t)
            
            self.update_mpc_constraints()
            
            self.dual_decomposition()
            
            self.update_dual_variables_trajectory()
            
            self.update_mpc_state_trajectories()
            
            
            
            
            
            
    