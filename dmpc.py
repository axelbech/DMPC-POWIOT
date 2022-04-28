import numpy as np
from datetime import datetime
# from pickle import dump
from json import dump
import os

class MPCsWrapper():
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
            
    def persist_results(self, path=''):
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        wrapper_type = type(self).__name__
        folder_name = f'{wrapper_type}-N{self.N}T{self.T}-{time}'
        os.mkdir(path + folder_name)
        for mpc in self.mpcs.values():
            mpc_type = type(mpc).__name__
            mpc_name = mpc.name
            f_name = f'{mpc_type}-{mpc_name}.json'
            mpc_dict = dict(traj_full=mpc.traj_full, params=mpc.params)
            with open(path + folder_name + '/' + f_name, 'w') as file:
                dump(mpc_dict, file, indent=4)
                # dump(mpc_dict, file)
        return folder_name
            
    def run_full(self):
        
        for t in range(self.T):
            print(f'time step {t}')
            
            self.update_mpc_parameters(t) # prepare mpc params
            
            self.update_mpc_constraints() # prepare mpc start constraints
            
            for mpc in self.mpcs.values():
                
                w_opt, f_opt = mpc.solve_optimization()
                
                mpc.set_optimal_state(mpc.w(w_opt))
            
            
            self.update_mpc_state_trajectories()
            
            self.update_mpc_initial_states()
            
            
class DistributedMPC(MPCsWrapper):

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
        
        self.f_sum_last = 1e6
    
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
        dual_variables_traj = np.empty((self.T,self.N-1 + self.T))
        dual_variables_traj[:] = np.nan
        return dual_variables_traj
    
    def iterate_dual_variables(self):
        """Prepare dual variables for next time step
        """
        self.dual_variables[:self.N-2] = self.dual_variables[1:]
    
    def update_dual_variables_trajectory(self, t):
        """Update dual variables trajectory at given time with current dual 
        variables

        Args:
            t (int): time step
        """
        self.dual_variables_traj[t, t:t+self.N-1] = self.dual_variables
        
    def update_mpc_dual_variables(self):
        """Update dual variables in mpcs with current dual variable
        """
        for mpc in self.mpcs.values():
            mpc.update_parameters_generic(dual_variables=self.dual_variables)
            
    def persist_results(self, path=''):
        folder_name = super().persist_results(path)
        dv_list = self.dual_variables_traj.tolist()
        file_name = 'dv_traj.json'
        with open(path + folder_name + '/' + file_name, 'w') as file:
            dump(dv_list, file, indent=4)
            
        
        
    def dual_decomposition(self):
        it = 0
        maxIt = 20
        
        f_tol = 1e-1 * self.N
        dv_tol = 0.1
        
        while it < maxIt:
            
            f_sum = 0
            dual_updates = np.zeros_like(self.dual_variables)
            
            dual_variables_last = np.copy(self.dual_variables)
            
            self.update_mpc_dual_variables()
            
            for mpc in self.mpcs.values():
                
                w_opt, f_opt = mpc.solve_optimization()
                f_sum += f_opt
                
                mpc.set_optimal_state(mpc.w(w_opt))
                mpc.set_initial_state(mpc.w(w_opt))
                dual_updates += mpc.get_dual_update_contribution()
                
            dual_updates += self.dual_update_constant
            dual_update_step_size = 20 / np.sqrt(1+it)
            dual_updates *= dual_update_step_size
            self.dual_variables += dual_updates
            self.dual_variables[self.dual_variables < 0] = 0
            
            f_diff = np.abs(f_sum - self.f_sum_last)
            self.f_sum_last = f_sum
            dv_diff = (np.abs(self.dual_variables-dual_variables_last)).mean()
            
            print(
                f'dual decomp iteration {it} '
                f'f_diff = {f_diff} '
                f'dual diff = {dv_diff}'
                )
            
            if f_diff < f_tol and dv_diff < dv_tol:
                break
            
            it += 1
        
        
    def run_full(self):
        
        for t in range(self.T):
            print(f'time step {t}')
            
            self.update_mpc_parameters(t) # prepare mpc params
            
            self.update_mpc_constraints() # prepare mpc start constraints
            
            self.dual_decomposition() # serial DD algorithm, get opt DVs
            
            self.update_dual_variables_trajectory(t)
            
            self.iterate_dual_variables()
            
            self.update_mpc_state_trajectories()
            
            self.update_mpc_initial_states()
            