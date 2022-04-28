#%%
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mpc import MPCCentralizedHomePeak, MPCCentralizedHomePeakQuadratic, MPCPeakStateDistributedQuadratic, MPCSingleHomeDistributed, MPCPeakStateDistributed
from dmpc import MPCsWrapper, DistributedMPC

N = 100
T = 100

n_houses = 2
p_max = 1.5

# outdoor_temperature = [10 for _ in range(N+T)]
# reference_temperature = [23 for _ in range(N+T)]
reference_temperature = list(np.fromfunction(
    lambda x: 5 * np.sin(x/10) + 22, (N+T,)
    ))
ref_temp_fixed = 22 * np.ones((N+T,))
# def price_func_exp(x):
#     return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
#             + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))
# spot_price = list(np.fromfunction(price_func_exp, (N+T,))) # Spot prices for two days, 5 min intervals

with open(r'data\spotdata\spot_price_5m.json', 'r') as file:
    spot_price = json.load(file)
    
with open('data/housedata/outdoor_temp_5m.json', 'r') as file:
    outdoor_temperature = json.load(file)


params = {
    'axel': {
        'opt_params': {
            'energy_weight': 5,
            'comfort_weight': 1,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed-2), # reference_temperature,
            'spot_price': spot_price,
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
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed+2), #reference_temperature,
            'spot_price': spot_price,
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
            'max_total_power': n_houses * p_max
        },
        'initial_state': {}
    }
}

#%%
cmpc_quad = MPCCentralizedHomePeakQuadratic(N, 'cent_quad', params)
w_quad = MPCsWrapper(N, T, {'centralized': cmpc_quad})
w_quad.run_full()
tj = cmpc_quad.traj_full

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
#     axel = MPCSingleHomeDistributed(N, 'axel', params['axel']),
#     seb =  MPCSingleHomeDistributed(N, 'seb', params['seb']),
#     peak = MPCPeakStateDistributedQuadratic(N, 'peak', params['peak'])
#     )
# dmpc_quad = DistributedMPC(N, T, mpcs_quad, 0)
# dmpc_quad.run_full()
# # dmpc_quad.persist_results('data/runs/')

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


#%%
# 

# 
# 