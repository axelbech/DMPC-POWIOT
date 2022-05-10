import numpy as np
from datetime import datetime
# from pickle import dump
import json
import os
import time
from multiprocessing import Manager, Process


class MPCWrapper():
    def __init__(
        self,
        N: int,
        T: int,
        controllers: list
        ):
        """Creates a wrapper for centralized or decentralized MPCs

        Args:
            N (int): mpc prediction horizon
            T (int): time steps
            controllers (list): list of controller objects
        """
        self.N = N
        self.T = T
        self.controllers = controllers
        
        self.manager = Manager()
        self.controller_results = {
            ctrl.name: self.manager.dict() for ctrl in controllers
            }
        
    def run_full(self):
        time_start = time.time()
        
        process_list = []
        for controller in self.controllers:
            process = Process(
                target=controller.run_full, 
                args=(self.controller_results[controller.name],)
                )
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()
            
        time_end = time.time()
        run_time_seconds = time_end - time_start
        print(f'Running time = {run_time_seconds} seconds')
        self.run_time_seconds = run_time_seconds
    
    def persist_results(self, path=''):
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        wrapper_type = type(self).__name__
        folder_name = f'{wrapper_type}-N{self.N}T{self.T}-{time}'
        os.mkdir(path + folder_name)
        with open(path + folder_name + '/run_time_seconds.json', 'w') as file:
            json.dump(self.run_time_seconds, file, indent=4)
        for mpc in self.controllers:
            mpc_type = type(mpc).__name__
            mpc_name = mpc.name
            f_name = f'{mpc_type}-{mpc_name}.json'
            mpc_dict = dict(self.controller_results[mpc_name]) #dict(traj_full=mpc.traj_full, params=mpc.params)
            with open(path + folder_name + '/' + f_name, 'w') as file:
                json.dump(mpc_dict, file, indent=4)
                # json.dump(mpc_dict, file)
        return folder_name
    

class DMPCWrapper(MPCWrapper):
    def __init__(
        self, 
        N: int,
        T: int, 
        controllers: list,
        coordinator,
        dual_variables_length: int
        ):
        super().__init__(N, T, controllers) 
        self.dual_variables_length = dual_variables_length
        self.coordinator = coordinator
        self.coordinator_results = self.manager.dict()
        self.coordination_dict = self.get_coordination_dict()
        
    def get_coordination_dict(self):
        public_coordination = self.manager.dict(
            dual_variables=np.zeros((self.dual_variables_length,)),
            t=None
        )
        private_coordination = self.manager.dict() # Nested managed dict
        for controller in self.controllers:
            controller_dict = self.manager.dict(
                dual_update_contribution=None,
                f_opt=None
            )
            private_coordination[controller.name] = controller_dict
            
        coordination_dict = dict(
            public=public_coordination,
            private=private_coordination
        )
        return coordination_dict
    
    def run_full(self):
        time_start = time.time()
        
        process_list = []
        process = Process(
            target=self.coordinator.run_full,
            args=(
                self.coordinator_results,
                self.coordination_dict['public'],
                self.coordination_dict['private']
            )
        )
        print(f'starting dmpc coordinator')
        process.start()
        process_list.append(process)
        
        for controller in self.controllers:
            process = Process(
                target=controller.run_full, 
                args=(
                    self.controller_results[controller.name],
                    self.coordination_dict['public'],
                    self.coordination_dict['private'][controller.name]
                    )
                )
            print(f'starting controller {controller.name}')
            process.start()
            process_list.append(process)
            
        for process in process_list:
            process.join()
            
        time_end = time.time()
        run_time_seconds = time_end - time_start
        print(f'Running time = {run_time_seconds} seconds')
        self.run_time_seconds = run_time_seconds
        
    def persist_results(self, path=''):
        folder_name = super().persist_results(path)
        coordinator_type = type(self.coordinator).__name__
        file_name = coordinator_type + '.json'
        with open(path + folder_name + '/' + file_name, 'w') as file:
            json.dump(dict(self.coordinator_results), file, indent=4)


class DMPCCoordinator():
    def __init__(
        self,
        N: int,
        T: int,
        controllers: list,
        dual_update_constant: float,
        dual_variables_length: int
        ):
        """Create a DMPC coordinator object

        Args:
            N (int): mpc prediction horizon
            T (int): time steps
            controllers (list): names of controllers
            dual_update_constant (float): constant value used in calculating 
            the dual update
        """
        self.N = N
        self.T = T
        self.controllers = controllers
        self.dual_update_constant = dual_update_constant
        self.dual_variables_length = dual_variables_length
        
        self.dual_variables = self.get_dual_variables()
        self.dual_variables_traj = self.get_dual_variables_trajectory()
        
    def get_dual_variables(self):
        """Builds dual variables

        Returns:
            ndarray: dual variables
        """
        return np.zeros(self.dual_variables_length)
    
    def get_dual_variables_trajectory(self):
        """Builds the dual varaible trajectory

        Returns:
            ndarray: dual variable trajectory
        """
        dual_variables_traj = np.empty((self.T,self.dual_variables_length + self.T))
        dual_variables_traj[:] = np.nan
        return dual_variables_traj
    
    def iterate_dual_variables(self):
        """Prepare dual variables for next time step
        """
        self.dual_variables[:self.dual_variables_length-1] = self.dual_variables[1:]
    
    def update_dual_variables_trajectory(self, t):
        """Update dual variables trajectory at given time with current dual 
        variables

        Args:
            t (int): time step
        """
        self.dual_variables_traj[t, t:t+self.dual_variables_length] = self.dual_variables

    def run_full(
        self,
        return_dict: dict,
        public_coordination: dict,
        private_coordination: dict
        ):
        """Run the full simulation

        Args:
            return_dict (dict): for returning dual variable trajectory
            public_coordination (dict): to coordinate time and dual variable
            private_coordination (dict): to coordinate dual update 
            contribution and optimal cost function value, contains multiple 
            managed private controller dicts
        """
        
        
        maxIt = 30
        
        self.f_sum_last = 1e6
        f_tol = 1e-3 * self.N
        dv_tol = 1e-3
        
        t = 0
        public_coordination['t'] = t
        # for t in range(self.T):
        while t < self.T:
            public_coordination['t'] = t
            public_coordination['dual_variables'] = self.dual_variables # If special start values
            print(f'\n\nstarting dual decomp at time step {t}, pid = {os.getpid()}')
            it = 0
            while it < maxIt:
                
                for controller in self.controllers:
                    private_coordination[controller]['f_opt'] = None
                
                while True:
                    is_all_controllers_ready = True
                    for controller in self.controllers:
                        if not private_coordination[controller]['f_opt']:
                            is_all_controllers_ready = False
                    if is_all_controllers_ready:
                        break
                    time.sleep(0.05)
                
                f_sum = 0
                dual_updates = np.zeros_like(self.dual_variables)
                dual_variables_last = np.copy(self.dual_variables)
                for controller in self.controllers:
                    f_sum += private_coordination[controller]['f_opt']
                    dual_updates += private_coordination[controller]['dual_update_contribution']
                    
                dual_updates += self.dual_update_constant
                dual_update_step_size = 0.15 # 2 / np.sqrt(1+it) 
                dual_updates *= dual_update_step_size
                self.dual_variables += dual_updates
                self.dual_variables[self.dual_variables < 0] = 0
                
                public_coordination['dual_variables'] = self.dual_variables
                
                f_diff = np.abs(f_sum - self.f_sum_last)
                self.f_sum_last = f_sum
                dv_diff = (np.abs(self.dual_variables-dual_variables_last)).mean()
                print(
                f'dual decomp iteration {it} , time step {t}, '
                f'f_diff = {f_diff.flatten()} '
                f'dual diff = {round(dv_diff,4)} '
                f'dv0 = {round(self.dual_variables[0],4)} '
                )
                
                if f_diff < f_tol and dv_diff < dv_tol:
                    break
                
                it += 1
                
            self.update_dual_variables_trajectory(t)
            self.iterate_dual_variables()
            public_coordination['dual_variables'] = self.dual_variables
            
            t += 1
            public_coordination['t'] = t
            
        for controller in self.controllers: # For terminating mpcs
            private_coordination[controller]['f_opt'] = None
            
        return_dict['dv_traj'] = self.dual_variables_traj.tolist()
            

class MPCWrapperSerial():
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
                json.dump(mpc_dict, file, indent=4)
                # json.dump(mpc_dict, file)
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
            
            
class DMPCWrapperSerial(MPCWrapperSerial):

    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict,
        dual_update_constant: float,
        dual_variables_length: int
        ):
        
        super().__init__(N, T, mpcs)
        self.dual_variables_length = dual_variables_length
        self.dual_variables = self.get_dual_variables()
        self.dual_variables_traj = self.get_dual_variables_trajectory()
        self.dual_update_constant = dual_update_constant
        
        self.f_sum_last = 1e6
    
    def get_dual_variables(self):
        """Builds dual variables

        Returns:
            ndarray: dual variables
        """
    #     return np.array([1.19259146e+01, 1.02966181e+01, 8.67673911e+00, 7.22517164e+00,
    #    5.86609441e+00, 4.66012486e+00, 3.60996313e+00, 2.71748427e+00,
    #    1.98811960e+00, 1.44316455e+00, 1.16754155e+00, 9.48234419e-01,
    #    7.48845008e-01, 5.69367301e-01, 4.09825887e-01, 3.96850465e-01,
    #    2.78072374e-01, 1.78606017e-01, 9.82982378e-02, 3.78909828e-02,
    #    6.95153734e-04, 5.67024178e-08, 4.06133252e-08, 3.42421441e-08,
    #    3.06327286e-08, 2.79391685e-08, 2.05145103e-08, 6.05080180e-08,
    #    3.40760704e-08, 3.17998452e-08, 3.06076766e-08, 3.00102460e-08,
    #    2.97992269e-08, 2.99065590e-08, 3.03665079e-08, 3.13342664e-08,
    #    3.31828249e-08, 3.68187924e-08, 1.80845452e-07, 1.55690903e-08,
    #    2.28578396e-08, 2.41726596e-08, 2.51374890e-08, 2.59103695e-08,
    #    2.66544038e-08, 2.75601986e-08, 2.89082990e-08, 3.12257725e-08,
    #    3.57772316e-08, 6.25740404e-08, 2.71315489e-02, 1.15478558e-08,
    #    1.89033426e-08, 2.08516414e-08, 2.28388715e-08, 2.54093049e-08,
    #    2.97793831e-08, 3.99390722e-08, 3.20203733e-07, 4.28398388e-02,
    #    1.20855532e-01, 2.33787171e-01, 4.03911091e-01, 6.51847332e-09,
    #    8.64706643e-09, 1.26476499e-08, 1.59067970e-08, 1.95629528e-08,
    #    2.48085939e-08, 3.63283677e-08, 1.28715523e-03, 5.78004213e-02,
    #    1.49774419e-01, 2.78152399e-01, 4.66683428e-01, 6.26473547e-09,
    #    8.36364012e-09, 1.12278037e-08, 1.53692174e-08, 1.88311314e-08,
    #    2.21544460e-08, 2.55050310e-08, 2.95370606e-08, 3.60171342e-08,
    #    6.27833773e-08, 1.76701929e-02, 1.04874956e-01, 6.66993724e-09,
    #    8.71164638e-09, 8.89543034e-09, 8.78448386e-09, 8.37538154e-09,
    #    7.58706072e-09, 6.26473535e-09, 6.26473357e-09, 6.26473346e-09,
    #    6.26473343e-09, 6.26473341e-09, 6.26473340e-09])
        return np.zeros(self.dual_variables_length)
    
    def get_dual_variables_trajectory(self):
        """Builds the dual varaible trajectory

        Returns:
            ndarray: dual variable trajectory
        """
        dual_variables_traj = np.empty((self.T,self.dual_variables_length + self.T))
        dual_variables_traj[:] = np.nan
        return dual_variables_traj
    
    def iterate_dual_variables(self):
        """Prepare dual variables for next time step
        """
        self.dual_variables[:self.dual_variables_length-1] = self.dual_variables[1:]
    
    def update_dual_variables_trajectory(self, t):
        """Update dual variables trajectory at given time with current dual 
        variables

        Args:
            t (int): time step
        """
        self.dual_variables_traj[t, t:t+self.dual_variables_length] = self.dual_variables
        
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
            json.dump(dv_list, file, indent=4)
            
    def project_dual_variables(self):
        """Project the computed dual variable onto its feasible plane. To be 
        used in the dual decomposition algorithm
        """
        self.dual_variables[self.dual_variables < 0] = 0
        
    def dual_decomposition(self):
        it = 0
        maxIt = 30
        
        f_tol = 1e-3 * self.N
        dv_tol = 1e-3
        
        while it < maxIt:
            # w0 = list(self.mpcs.values())[0].w0.master[0]
            # print(f'w0 = {w0}')
            f_sum = 0
            dual_updates = np.zeros_like(self.dual_variables)
            dual_variables_last = np.copy(self.dual_variables)
            self.update_mpc_dual_variables()
            
            for mpc in self.mpcs.values():
                
                w_opt, f_opt = mpc.solve_optimization()
                f_sum += f_opt
                
                mpc.set_optimal_state(mpc.w(w_opt))
                mpc.set_initial_state(mpc.w(w_opt))
                # print(mpc.name, ' peak0 = ', mpc.w_opt['peak_state',1])
                dual_updates += mpc.get_dual_update_contribution()
                
            dual_updates += self.dual_update_constant
            dual_update_step_size = 2 / np.sqrt(1+it) # alpha value
            dual_updates *= dual_update_step_size # alpha * residual
            self.dual_variables += dual_updates
            self.project_dual_variables()
            # self.dual_variables[self.dual_variables < 0] = 0 # Project DV
            
            f_diff = np.abs(f_sum - self.f_sum_last)
            self.f_sum_last = f_sum
            dv_diff = (np.abs(self.dual_variables-dual_variables_last)).mean()
            
            print(
                f'dual decomp iteration {it} '
                f'f_diff = {f_diff.flatten().flatten()} '
                f'dual diff = {round(dv_diff,4)} '
                # f'dv = {self.dual_variables}'
                f'dv = {round(self.dual_variables[0],2)} '
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
            
            
class DMPCWrapperSerialProxGrad(DMPCWrapperSerial):
    
    def __init__(
        self, 
        N: int, 
        T: int, 
        mpcs: dict,
        dual_update_constant: float, 
        dual_variables_length: int, proximalGradientSolver):
        super().__init__(N, T, mpcs, dual_update_constant, dual_variables_length)
        
        self.proximalGradientSolver = proximalGradientSolver
    
    def project_dual_variables(self):
        self.proximalGradientSolver.update_parameters_generic(
            mu_plus = self.dual_variables
        )
        dual_variables = self.proximalGradientSolver.solve_optimization()
        self.dual_variables = np.array(dual_variables).flatten()
