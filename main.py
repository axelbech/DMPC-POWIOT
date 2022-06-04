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
    MPCCentralized,
    MPCCentralizedHomeFixed,
    MPCCentralizedHomePeak, 
    MPCCentralizedHomePeakQuadratic, 
    MPCCentralizedHomeSinglePeak,
    MPCCentralizedSinglePeakConvex,
    MPCCentralizedHourly,
    MPCPeakStateDistributed,
    MPCPeakStateDistributedQuadratic, 
    MPCHourlyPeakDistributed,
    MPCSingleHome, 
    MPCSingleHomeDistributed, 
    MPCSingleHomePeak,
    MPCSingleHomePeakDistributed,
    MPCSingleHomeHourlyDistributed,
    MPCSinglePeakDistributed,
    ProximalGradientSolver,
)
from wrappers import (
    MPCWrapper, 
    DMPCWrapper,
    MPCWrapperSerial, 
    DMPCWrapperSerial,
    DMPCCoordinator,
    DMPCCoordinatorProxGrad,
    DMPCWrapperSerialProxGrad,
)

N = 288
T = 288


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
with open(r'data\power\pwr_ext_5m_1119.json', 'r') as file:
    pwr_1119 = json.load(file)
with open(r'data\power\pwr_ext_5m_1121.json', 'r') as file:
    pwr_1121 = json.load(file)
with open(r'data\power\pwr_ext_5m_1123.json', 'r') as file:
    pwr_1123 = json.load(file)
with open(r'data\power\pwr_ext_5m_1125.json', 'r') as file:
    pwr_1125 = json.load(file)
with open(r'data\power\pwr_ext_5m_1127.json', 'r') as file:
    pwr_1127 = json.load(file)
with open(r'data\power\pwr_ext_5m_1129.json', 'r') as file:
    pwr_1129 = json.load(file)
with open(r'data\power\pwr_ext_5m_1205.json', 'r') as file:
    pwr_1205 = json.load(file)
with open(r'data\power\pwr_ext_5m_1208.json', 'r') as file:
    pwr_1208 = json.load(file)
pwr_mult = 0.33
ext_power_avg = (np.array(ext_power_avg) * pwr_mult).tolist() # 0.33 instead of 0.6 to emulate 1/3 size home
pwr_1119 = (np.array(pwr_1119) * pwr_mult).tolist()
pwr_1121 = (np.array(pwr_1121) * pwr_mult).tolist()
pwr_1123 = (np.array(pwr_1123) * pwr_mult).tolist()
pwr_1125 = (np.array(pwr_1125) * pwr_mult).tolist()
pwr_1127 = (np.array(pwr_1127) * pwr_mult).tolist()
pwr_1129 = (np.array(pwr_1129) * pwr_mult).tolist()
pwr_1205 = (np.array(pwr_1205) * pwr_mult).tolist()
pwr_1208 = (np.array(pwr_1208) * pwr_mult).tolist()
ext_power_none = np.zeros((N+T,)).tolist()
ext_pwr_list = [pwr_1127,pwr_1129,pwr_1119,pwr_1121,pwr_1123,pwr_1125,pwr_1205,pwr_1208]
with open('data/housedata/outdoor_temp_5m.json', 'r') as file:
    outdoor_temperature = json.load(file)

n_houses = 2
peak_weight = n_houses * 0.25
peak_weight_quad = n_houses * 0.01 # 0.02
max_total_power = n_houses * 0.2

params = {
    'House1': {
        'opt_params': {
            'energy_weight': 0.01,
            'comfort_weight': 0.05,
            'slack_min_weight': 1,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 2.5, #Colder outside  3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed), # reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': pwr_1127, # ext_power_none,# ext_power_avg,# pwr_1127,
            'ext_power_avg': ext_power_avg # ext_power_none, # ext_power_avg
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
            'comfort_weight': 0.05,
            'slack_min_weight': 1,
            'rho_out': 0.018,
            'rho_in': 0.37,
            'COP': 2.5, #Colder outside 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': list(ref_temp_fixed), #reference_temperature,
            'min_temperature': list(min_temp),
            'spot_price': spot_price,
            'ext_power_real': pwr_1129, # ext_power_none,# ext_power_avg, # pwr_1129,
            'ext_power_avg': ext_power_avg,# ext_power_none# ext_power_avg
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
            'peak_weight_quad': peak_weight_quad,
        },
        'bounds': {
            # 'max_total_power': n_houses * 10
        },
        'initial_state': {
            'peak_state': 0
        }
    },
}

params_localized = copy.copy(params)
del params_localized['peak']
for house in params_localized:
    params_localized[house]['opt_params']['peak_weight'] = peak_weight
    params_localized[house]['initial_state']['peak_state'] = 0
    
peak_weight_single = 10  #peak_weight * N # 0.5 * N
peak_weight_quad_single = 1 # peak_weight_quad * N # 0.5 * N
params_single_peak = copy.copy(params)
params_single_peak['peak']['opt_params']['peak_weight'] = peak_weight_single
params_single_peak['peak']['opt_params']['peak_weight_quad'] = peak_weight_quad_single

params_8 = copy.copy(params_single_peak)
for i in range(3,9):
    house = "House" + str(i)
    params_8[house] = copy.deepcopy(params_8['House1'])
    params_8[house]['opt_params']['ext_power_real'] = ext_pwr_list[i-1]
    
    
hourly_weight_quad = 0.1 * 12 # 0.5 * 12 # peak_weight_quad_single * 12
params_hourly = copy.copy(params)
del params_hourly['peak']
params_hourly['hourly_peak'] = {'opt_params': {'hourly_weight_quad': hourly_weight_quad,},
        'initial_state': { 'hourly_peak': 0}}

params_hourly_8 = copy.copy(params_8)
del params_hourly_8['peak']
params_hourly_8['hourly_peak'] = {'opt_params': {'hourly_weight_quad': hourly_weight_quad,},
        'initial_state': { 'hourly_peak': 0}}


# if __name__ == '__main__':    # FOR RUNNING THE HOURLY PEAK COST, 2 HOUSES
#     mpcs = dict(
#     House1 = MPCSingleHomeHourlyDistributed(N, T, 'House1', params_hourly['House1']),
#     House2 =  MPCSingleHomeHourlyDistributed(N, T, 'House2', params_hourly['House2']),
#     hourly_peak = MPCHourlyPeakDistributed(N, T, 'hourly_peak', params_hourly['hourly_peak'])
#     )
#     wrapper = DMPCWrapperSerial(N, T, mpcs, 0, dual_variables_length=int(np.ceil(N/12)), step_size=0.15)
#     wrapper.run_full()  
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':    
#     cmpc = MPCCentralizedHourly(N, T, 'cent', params_hourly, int(np.ceil(N/12)))
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':      # FOR RUNNING THE QUADRATIC PEAK COST, 2 HOUSES
#     mpcs = dict(
#     House1 = MPCSingleHomeDistributed(N, T, 'House1', params_single_peak['House1']),
#     House2 =  MPCSingleHomeDistributed(N, T, 'House2', params_single_peak['House2']),
#     peak = MPCSinglePeakDistributed(N, T, 'peak', params['peak'])
#     )
#     wrapper = DMPCWrapperSerial(N, T, mpcs, 0, dual_variables_length=N-1, step_size=0.1)
#     wrapper.run_full()  
#     wrapper.persist_results('data/runs/')
    
# if __name__ == '__main__':    
#     cmpc = MPCCentralizedSinglePeakConvex(N, T, 'cent', params_single_peak, N-1)
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
    

# if __name__ == '__main__':    # FOR RUNNING THE LINEAR PEAK COST, 2 HOUSES
#     proxGradSolver = ProximalGradientSolver(N, peak_weight_single)
#     mpcs = dict(
#     House1 = MPCSingleHomeDistributed(N, T, 'House1', params['House1']),
#     House2 =  MPCSingleHomeDistributed(N, T, 'House2', params['House2'])
#     )
#     wrapper = DMPCWrapperSerialProxGrad(N, T, mpcs, 0, 
#                                         dual_variables_length=N-1,
#                                         step_size = 0.5,
#                                         proximalGradientSolver=proxGradSolver)
#     wrapper.run_full()  
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':    
#     cmpc = MPCCentralizedHomeSinglePeak(N, T, 'cent', params_single_peak, N-1)
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':    # FOR RUNNING THE DECENTRALIZED APPROACH, 2H
#     mpcs = dict(
#         House1 = MPCSingleHome(N, T, 'House1', params_single_peak['House1']),
#         House2 =  MPCSingleHome(N, T, 'House2', params_single_peak['House2'])
#         )
#     wrapper = MPCWrapper(N, T, list(mpcs.values()))
#     # wrapper = MPCWrapperSerial(N, T, mpcs)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

#################### 8 houses #######################

# if __name__ == '__main__':    # QUAD SINGLE PEAK 8H
#     cmpc = MPCCentralizedSinglePeakConvex(N, T, 'cent', params_8, N-1)
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
    
# if __name__ == '__main__':
#     mpcs = {}
#     for i in range(1,9):
#         mpcname = 'House' + str(i)
#         mpcs[mpcname] = MPCSingleHomeDistributed(N, T, mpcname, params_8[mpcname])
#     mpcs['peak'] = MPCSinglePeakDistributed(N, T, 'peak', params_8['peak'])
#     coordinator = DMPCCoordinator(N, T, [ctrl for ctrl in mpcs], dual_update_constant=0, dual_variables_length=N-1, step_size=0.05)
#     wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs.values()],coordinator, dual_variables_length=N-1)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':    # LINEAR PROJECTED SINGLE PEAK 8H
#     cmpc = MPCCentralizedHomeSinglePeak(N, T, 'cent', params_8, N-1)
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
    
# if __name__ == '__main__':
#     mpcs = {}
#     for i in range(1,9):
#         mpcname = 'House' + str(i)
#         mpcs[mpcname] = MPCSingleHomeDistributed(N, T, mpcname, params_8[mpcname])
#     proxGradSolver = ProximalGradientSolver(N, peak_weight_single)
#     coordinator = DMPCCoordinatorProxGrad(N, T, [ctrl for ctrl in mpcs],
#      dual_update_constant=0, dual_variables_length=N-1, step_size=0.1,
#      proximalGradientSolver=proxGradSolver)
#     wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs.values()],coordinator, dual_variables_length=N-1)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':    # HOURLY PEAK COST 8H
#     cmpc = MPCCentralizedHourly(N, T, 'cent', params_hourly_8, int(np.ceil(N/12)))
#     wrapper = MPCWrapper(N, T, [cmpc])
#     # wrapper = MPCWrapperSerial(N, T, dict(cent=cmpc))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':
#     mpcs = {}
#     for i in range(1,9):
#         mpcname = 'House' + str(i)
#         mpcs[mpcname] = MPCSingleHomeHourlyDistributed(N, T, mpcname, params_hourly_8[mpcname])
#     mpcs['hourly_peak'] = MPCHourlyPeakDistributed(N, T, 'hourly_peak', params_hourly_8['hourly_peak'])
#     coordinator = DMPCCoordinator(N, T, [ctrl for ctrl in mpcs], dual_update_constant=0, dual_variables_length=int(np.ceil(N/12)), step_size=0.15)
#     wrapper = DMPCWrapper(N, T, [ctrl for ctrl in mpcs.values()],coordinator, dual_variables_length=int(np.ceil(N/12)))
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__': #  DECENTRALIZED APPROACH, 8H
#     mpcs = {}
#     for i in range(1,9):
#         mpcname = 'House' + str(i)
#         mpcs[mpcname] = MPCSingleHome(N, T, mpcname, params_8[mpcname])
#     wrapper = MPCWrapper(N, T, list(mpcs.values()))
#     # wrapper = MPCWrapperSerial(N, T, mpcs)
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')


# if __name__ == '__main__':
#     mpcs = dict(
#     House1 = MPCSingleHomeDistributed(N, T, 'House1', params['House1']),
#     House2 =  MPCSingleHomeDistributed(N, T, 'House2', params['House2'])
#     )
#     wrapper = DMPCWrapperSerial(N, T, mpcs, -max_total_power, dual_variables_length=N-1)
#     wrapper.run_full()  
#     wrapper.persist_results('data/runs/')
    
# if __name__ == '__main__':
#     cmpc = MPCCentralizedHome(N, T, 'cent', params)
#     wrapper = MPCWrapper(N, T, [cmpc])
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')

# if __name__ == '__main__':
#     mpcs = dict(
#         House1 = MPCSingleHome(N, T, 'House1', params['House1']),
#         House2 =  MPCSingleHome(N, T, 'House2', params['House2'])
#         )
#     wrapper = MPCWrapper(N, T, [ctrl for ctrl in mpcs.values()])
#     wrapper.run_full()
#     wrapper.persist_results('data/runs/')
    
    
# if __name__ == '__main__':
#     mpcs = dict(
#     House1 = MPCSingleHomePeakDistributed(N, T, 'House1', params_localized['House1']),
#     House2 =  MPCSingleHomePeakDistributed(N, T, 'House2', params_localized['House2'])
#     )
#     wrapper = DMPCWrapperSerial(N, T, mpcs, 0, dual_variables_length=N-1)
#     wrapper.run_full()  
#     wrapper.persist_results('data/runs/')
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
#     cmpc_lin = MPCCentralizedHomePeak(N, T, 'cent_lin', params)
#     wrapper = MPCWrapper(N, T, [cmpc_lin])
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

