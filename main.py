#%%
import json
from multiprocessing import Process, Manager
from concurrent.futures import ProcessPoolExecutor
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpc import (
    MPCCentralizedHomePeak, 
    MPCCentralizedHomePeakQuadratic, 
    MPCPeakStateDistributed,
    MPCPeakStateDistributedQuadratic, 
    MPCSingleHome, 
    MPCSingleHomeDistributed, 
    MPCSingleHomePeak,
    MPCSingleHomePeakDistributed,
)
from wrappers import (
    MPCWrapper, 
    DMPCWrapper,
    MPCWrapperSerial, 
    DMPCWrapperSerial,
    DMPCCoordinator,
)

N = 100
T = 1

n_houses = 2
p_max = 1.5

# outdoor_temperature = [10 for _ in range(N+T)]
# reference_temperature = [23 for _ in range(N+T)]
reference_temperature = list(np.fromfunction(
    lambda x: 5 * np.sin(x/10) + 22, (N+T,)
    ))
ref_temp_fixed = 22 * np.ones((N+T,))

min_temp = 15 * np.ones((N+T,))
# def price_func_exp(x):
#     return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
#             + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))
# spot_price = list(np.fromfunction(price_func_exp, (N+T,))) # Spot prices for two days, 5 min intervals

with open(r'data\spotdata\spot_price_5m.json', 'r') as file:
    spot_price = json.load(file)
    
with open(r'data\power\pwr_ext_avg.json', 'r') as file:
    ext_power_avg = json.load(file)
with open(r'data\power\pwr_ext_5m_1129.json', 'r') as file:
    pwr_1129 = json.load(file)
with open(r'data\power\pwr_ext_5m_1127.json', 'r') as file:
    pwr_1127 = json.load(file)
ext_power_peak = (np.array(ext_power_avg) * 1.2).tolist()
ext_power_avg = (np.array(ext_power_avg) * 0.6).tolist()
pwr_1129 = (np.array(pwr_1129) * 0.6).tolist()
pwr_1127 = (np.array(pwr_1127) * 0.6).tolist()
ext_power_none = np.zeros((N+T,)).tolist()

with open('data/housedata/outdoor_temp_5m.json', 'r') as file:
    outdoor_temperature = json.load(file)

peak_weight = 0.5


params = {
    'House1': {
        'opt_params': {
            'energy_weight': 0.01,
            'comfort_weight': 0.05,
            'slack_min_weight': 1,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed-2), # reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': ext_power_none,# ext_power_avg,# pwr_1127,
            'ext_power_avg': ext_power_none, # ext_power_avg
        },
        'bounds': {
            'P_max': p_max,
        },
        'initial_state': {
            'room_temp': 24,
            'wall_temp': 18,
        }
    },
    'House2': {
        'opt_params': {
            'energy_weight': 0.01,
            'comfort_weight': 0.1,
            'slack_min_weight': 1,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed+2), #reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': ext_power_none,# ext_power_avg, # pwr_1129,
            'ext_power_avg': ext_power_none# ext_power_avg
        },
        'bounds': {
            'P_max': p_max,
        },
        'initial_state': {
            'room_temp': 20,
            'wall_temp': 15,
        }
    },
    'peak':{
        'opt_params': {
            'peak_weight': peak_weight,
            # 'ext_power_real': ext_power_peak, # QUICK FIX
            # 'ext_power_avg':  ext_power_peak
        },
        'bounds': {
            # 'max_total_power': n_houses * 10
        },
        'initial_state': {
            'peak_state': 0
        }
    }
}

params_localized = copy.copy(params)
del params_localized['peak']
for house in params_localized:
    params_localized[house]['opt_params']['peak_weight'] = peak_weight
    params_localized[house]['initial_state']['peak_state'] = 0
    
    
if __name__ == '__main__':
    mpcs = dict(
    House1 = MPCSingleHomePeakDistributed(N, T, 'House1', params_localized['House1']),
    House2 =  MPCSingleHomePeakDistributed(N, T, 'House2', params_localized['House2'])
    )
    wrapper = DMPCWrapperSerial(N, T, mpcs, 0, dual_variables_length=N-1)
    wrapper.run_full()  
    # wrapper.persist_results('data/runs/')
    # coordinator = DMPCCoordinator(N, T, [ctrl for ctrl in mpcs], dual_update_constant=0, dual_variables_length=N-1)
    # wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs.values()],coordinator, dual_variables_length=N-1)
    # wrapper.run_full()
    # wrapper.persist_results('data/runs/')
    
# if __name__ == '__main__':
#     mpcs = dict(
#     House1 = MPCSingleHomePeak(N, T, 'House1', params_localized['House1']),
#     House2 =  MPCSingleHomePeak(N, T, 'House2', params_localized['House2'])
#     )
#     wrapper = MPCWrapper(N, T, [ctrl for ctrl in mpcs.values()])
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':
#     cmpc_quad = MPCCentralizedHomePeakQuadratic(N, T, 'cent_quad', params)
#     wrapper = MPCWrapper(N, T, [cmpc_quad])
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
#%%
# if __name__ == '__main__':
#     mpcs = dict(
#         House1 = MPCSingleHomeDistributed(N, T, 'House1', params['House1']),
#         House2 =  MPCSingleHomeDistributed(N, T, 'House2', params['House2']),
#         peak = MPCPeakStateDistributedQuadratic(N, T, 'peak', params['peak'])
#         )
#     coordinator = DMPCCoordinator(N, T, [ctrl for ctrl in mpcs], 0, dual_variables_length=N-1)
#     wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs.values()], coordinator, dual_variables_length=N-1)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
    
#     wr = DistributedMPC(N, T, mpcs, 0)
#     wr.run_full()
#%%

    
# #%%
# if __name__ == '__main__':
#     mpcs = dict(
#         House1 = MPCSingleHome(N, T, 'House1', params['House1']),
#         House2 =  MPCSingleHome(N, T, 'House2', params['House2'])
#         )
#     wrapper = MPCWrapper(N, T, [ctrl for ctrl in mpcs.values()])
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

#%%

# if __name__ == '__main__':
#     mpc_seb = MPCSingleHome(N, T, 'seb', params['seb'])
#     M = 16
#     with ProcessPoolExecutor() as executor:
#         m = executor.map(mpc_seb.run_full, [{}]*M)
#         res = list(m)
#         print(m)


# if __name__ == '__main__':
#     mpc_seb = MPCSingleHome(N, T, 'seb', params['seb'])
#     results_manager = Manager()
#     res_seb = results_manager.dict()
#     process_list = []
#     for i in range(4):
#         print('creating process')
#         then = time.time()
#         process = Process(target=mpc_seb.run_full, args=(res_seb,))
#         process.start()
#         process_list.append(process)
#         now = time.time()
#         print(f'######### time diff = {now-then} #########')
#     for process in process_list:
#         process.join()
#     print(res_seb['traj_full']['room_temp'])
    

#%%
# cmpc_quad = MPCCentralizedHomePeakQuadratic(N, 'cent_quad', params)
# w_quad = MPCsWrapper(N, T, {'centralized': cmpc_quad})
# w_quad.run_full()
# tj = cmpc_quad.traj_full

#%%
# mpcs_lin = dict(
#     axel = MPCSingleHomeDistributed(N, 'axel', params['axel']),
#     seb =  MPCSingleHomeDistributed(N, 'seb', params['seb']),
#     peak = MPCPeakStateDistributed(N, 'peak', params['peak'])
#     )
# dmpc_lin = DistributedMPC(N, T, mpcs_lin, 0)
# dmpc_lin.run_full()
# # dmpc_lin.persist_results('data/runs/')

# #%%
# mpcs_quad = dict(
#     axel = MPCSingleHomeDistributed(N, T, 'axel', params['axel']),
#     seb =  MPCSingleHomeDistributed(N, T, 'seb', params['seb']),
#     peak = MPCPeakStateDistributedQuadratic(N, T, 'peak', params['peak'])
#     )
# dmpc_quad = DistributedMPC(N, T, mpcs_quad, 0)
# dmpc_quad.run_full()
# dmpc_quad.persist_results('data/runs/')

# #%%
# cmpc_lin = MPCCentralizedHomePeak(N, 'cent_lin', params)
# w_lin = MPCsWrapper(N, T, {'centralized': cmpc_lin})
# w_lin.run_full()
# # w_lin.persist_results('data/runs/')

# #%%
# cmpc_quad = MPCCentralizedHomePeakQuadratic(N, 'cent_quad', params)
# w_quad = MPCsWrapper(N, T, {'centralized': cmpc_quad})
# w_quad.run_full()
# w_quad.persist_results('data/runs/')

