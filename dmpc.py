import numpy as np

class PartitionedMPC():
    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict
        ):
        """Create a partitioned mpc object

        Args:
            N (int): optimization window length
            T (int): time steps
            mpcs (dict): mpc object, accessed by name
        """
        self.N = N
        self.T = T
        self.mpcs = mpcs
        
    def update_mpc_constraints(self):
        """Update the constraints before an optimization
        """
        for mpc in self.mpcs.values():
            mpc.update_constraints()
            
    def update_mpc_initial_states(self):
        """Update the initial state of each mpc with their optimal states after
        an optimization
        """
        for mpc in self.mpcs.values():
            mpc.update_initial_state()
        
    def update_mpc_state_trajectories(self):
        """Update the trajectories of each mpc with their newest values
        """
        for mpc in self.mpcs.values():
            mpc.update_trajectory()
            
    def update_mpc_parameters(self, t: int):
        """Update the time variant optimization parameters

        Args:
            t (int): time step
        """
        for mpc in self.mpcs.values():
            mpc.update_parameters(t)
            
            
class DistributedMPC(PartitionedMPC):

    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict,
        dual_update_constant: float
        ):
        
        super().__init__(N, T, mpcs)
        self.dual_variables = self.get_dual_variables()
        self.dual_variables_traj = self.get_dual_variables_trajectory()
        self.dual_update_constant = dual_update_constant
    
    def get_dual_variables(self):
        """Builds dual variables

        Returns:
            ndarray: dual variables
        """
        return np.zeros(self.N-1)
    
    def get_dual_variables_trajectory(self):
        """Builds the dual varaible trajectory

        Returns:
            ndarray: dual variable trajectory
        """
        return np.zeros((self.T,self.N-1))
    
    def update_dual_variables_trajectory(self, t):
        """Update dual variables trajectory at given time with current dual 
        variables

        Args:
            t (int): time step
        """
        self.dual_variables_traj[t, :] = self.dual_variables
        
    def update_mpc_dual_variables(self):
        """Update dual variables in mpcs with current dual variable
        """
        for mpc in self.mpcs.values():
            mpc.update_parameters_generic(dual_variables=self.dual_variables)
        
    def dual_decomposition(self):
        it = 0
        maxIt = 10
        
        f_tol = 1e-2 
        f_diff = 1e6
        f_sum_last = 1e6
        
        while it < maxIt and f_diff > f_tol:
            
            f_sum = 0
            dual_updates = np.zeros_like(self.dual_variables)
            
            for mpc in self.mpcs.values():
                
                w_opt, f_opt = mpc.solve_optimization()
                f_sum += f_opt
                
                mpc.update_optimal_state(mpc.w(w_opt))
                dual_updates += mpc.get_dual_update_contribution()
                
            dual_update_step_size = 20  / np.sqrt(1+it)
            self.dual_variables += dual_update_step_size * dual_updates
            self.update_mpc_dual_variables()
            
            f_diff = np.abs(f_sum - f_sum_last)
            f_sum_last = f_sum
            
            print(f'dual decomp iteration {it},  f_diff = {f_diff}')
            
            it += 1
        
        
    def run_full(self):
        
        for t in range(self.T):
            print(f'time step {t}')
            
            self.update_mpc_parameters(t) # prepare mpc params
            
            self.update_mpc_constraints() # prepare mpc start constraints
            
            self.dual_decomposition() # serial DD algorithm, get opt DVs
            
            self.update_dual_variables_trajectory(t)
            
            self.update_mpc_initial_states()
            
            self.update_mpc_state_trajectories()
            