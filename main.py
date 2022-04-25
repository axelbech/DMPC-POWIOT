#%%
from mpc import MPCCentralizedHomePeak, MPCPeakStateDistributedQuadratic, MPCSingleHomeDistributed, MPCPeakStateDistributed
from dmpc import MPCsWrapper, DistributedMPC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

N = 288
T = 50

n_houses = 1
p_max = 1.5

outdoor_temperature = [10 for _ in range(N+T)]
# reference_temperature = [23 for _ in range(N+T)]
reference_temperature = list(np.fromfunction(
    lambda x: 5 * np.sin(x/10) + 22, (N+T,)
    ))
def price_func_exp(x):
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))
spot_price = list(np.fromfunction(price_func_exp, (N+T,))) # Spot prices for two days, 5 min intervals


params = {
    'axel': {
        'opt_params': {
            'energy_weight': 80,
            'comfort_weight': 1,
            'rho_out': 0.18,
            'rho_in': 0.37,
            'COP': 3.5,
            'outdoor_temperature': outdoor_temperature,
            'reference_temperature': reference_temperature,
            'spot_price': spot_price,
        },
        'bounds': {
            'P_max': p_max,
        },
        'initial_state': {
            'room': 24,
            'wall': 18,
        }
    },
    'peak':{
        'opt_params': {
            'peak_weight': 10
        },
        'bounds': {
            'max_total_power': n_houses * p_max
        },
        'initial_state': {}
    }
}

# mpcc = MPCCentralizedHomePeak(N, 'centralized', params)
# cmpc_wrapper = MPCsWrapper(N, T, {'centralized': mpcc})
# cmpc_wrapper.run_full()
# cmpc_wrapper.save_mpcs_to_file('data/')

mpc_axel = MPCSingleHomeDistributed(N, 'axel', params['axel'])
mpc_peak = MPCPeakStateDistributedQuadratic(N, 'peak', params['peak'])
mpcs = dict(
    axel = mpc_axel,
    peak = mpc_peak
    )
dmpc = DistributedMPC(N, T, mpcs, 0)
dmpc.run_full()
dmpc.save_mpcs_to_file('data/runs/')

# pwr_c = mpcc.traj_full['axel']['P_hp']
# pwr_d = mpc_axel.traj_full['P_hp']
# figp, axp = plt.subplots()
# axp.plot(pwr_c, label='centralized')
# axp.plot(pwr_d, label='distributed')
# axp.legend()
# axp.set_title('Power usage [W]')

# room_c = mpcc.traj_full['axel']['room_temp']
# room_d = mpc_axel.traj_full['room']
# figr, axr = plt.subplots()
# axr.plot(room_c, label='centralized')
# axr.plot(room_d, label='distributed')
# axr.legend()
# axr.set_title('Room temperature')

# plt.show()


# #%%

# mpc_axel = mpcs['axel']

# mpc_peak = mpcs['peak']

# pwr = mpc_axel.traj_full['P_hp']
# figp, axp = plt.subplots()
# axp.plot(pwr)
# axp.legend()
# axp.set_title('Power usage [W]')

# room = mpc_axel.traj_full['room']
# figr, axr = plt.subplots()
# axr.plot(room)
# axr.legend()
# axr.set_title('Room temperature')

# wall = mpc_axel.traj_full['wall']
# figw, axw = plt.subplots()
# axw.plot(wall)
# axw.legend()
# axw.set_title('Wall temperature')

# wall = mpc_peak.traj_full['peak']
# fig, ax = plt.subplots()
# ax.plot(wall)
# ax.legend()
# ax.set_title('Peak state')


# traj_dv = dmpc.dual_variables_traj
# x, y = np.meshgrid(np.arange(traj_dv.shape[1]), np.arange(traj_dv.shape[0]))
# z = traj_dv
# figdv, axdv = plt.subplots(subplot_kw={"projection": "3d"})
# axdv.set_title('Dual variable values')
# surf = axdv.plot_surface(x, y, z, cmap=cm.coolwarm)
# axdv.zaxis.set_rotate_label(False)
# axdv.set_xlabel('Time')
# axdv.set_ylabel('Iteration')
# axdv.set_zlabel('$\lambda$')

# plt.show()
# %%
