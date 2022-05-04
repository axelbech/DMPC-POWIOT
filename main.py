#%%
import json
from multiprocessing import Process, Manager
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpc import MPCCentralizedHomePeak, MPCCentralizedHomePeakQuadratic, MPCPeakStateDistributedQuadratic, MPCSingleHome, MPCSingleHomeDistributed, MPCPeakStateDistributed
from wrappers import MPCWrapper, MPCsWrapper, DistributedMPC, DMPCWrapper, DMPCCoordinator

N = 288
T = 288

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
ext_power_avg = (np.array(ext_power_avg) * 0.6).tolist()
pwr_1129 = (np.array(pwr_1129) * 0.6).tolist()
pwr_1127 = (np.array(pwr_1127) * 0.6).tolist()

with open('data/housedata/outdoor_temp_5m.json', 'r') as file:
    outdoor_temperature = json.load(file)



params = {
    'axel': {
        'opt_params': {
            'energy_weight': 5,
            'comfort_weight': 1,
            'slack_min_weight': 10,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed-2), # reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': pwr_1127,
            'ext_power_avg': ext_power_avg
        },
        'bounds': {
            'P_max': p_max,
        },
        'initial_state': {
            'room_temp': 24,
            'wall_temp': 18,
        }
    },
    'seb': {
        'opt_params': {
            'energy_weight': 5,
            'comfort_weight': 1,
            'slack_min_weight': 10,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed+2), #reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': pwr_1129,
            'ext_power_avg': ext_power_avg
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
            'peak_weight': 20
        },
        'bounds': {
            'max_total_power': n_houses * 10
        },
        'initial_state': {}
    }
}

#%%
# if __name__ == '__main__':
#     mpcs_lin = dict(
#         axel = MPCSingleHomeDistributed(N, T, 'axel', params['axel']),
#         seb =  MPCSingleHomeDistributed(N, T, 'seb', params['seb']),
#         peak = MPCPeakStateDistributedQuadratic(N, T, 'peak', params['peak'])
#         )
#     coordinator = DMPCCoordinator(N, T, [ctrl for ctrl in mpcs_lin], 0)
#     wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs_lin.values()], coordinator)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
#%%

if __name__ == '__main__':
    cmpc_quad = MPCCentralizedHomePeakQuadratic(N, T, 'cent_quad', params)
    wrapper = MPCWrapper(N, T, [cmpc_quad])
    wrapper.run_full()
    wrapper.persist_results('data/runs/')
    
#%%
# if __name__ == '__main__':
#     mpcs = dict(
#         axel = MPCSingleHome(N, T, 'axel', params['axel']),
#         seb =  MPCSingleHome(N, T, 'seb', params['seb'])
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

