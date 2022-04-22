#%%
from mpc import MPCSingleHomeDistributed, MPCPeakStateDistributed
from dmpc import DistributedMPC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

N = 50
T = 50

n_houses = 1
p_max = 1.5

outdoor_temperature = [10 for _ in range(N+T)]
# reference_temperature = [23 for _ in range(N+T)]
reference_temperature = list(np.fromfunction(lambda x: 5 * np.sin(x/10) + 22, (N+T,)))
def price_func_exp(x):
    return (1 + 0.7 *np.exp(-((x-96)/40)**2) + np.exp(-((x-216)/60)**2)
            + 0.7 *np.exp(-((x-96-288)/40)**2) + np.exp(-((x-216-288)/60)**2))
spot_price = list(np.fromfunction(price_func_exp, (N+T,))) # Spot prices for two days, 5 min intervals


params = {
    'axel': {
        'opt_params': {
            'energy_weight': 100,
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
            'peak_weight': 20
        },
        'bounds': {
            'max_total_power': n_houses * 0.2* p_max
        },
        'initial_state': {}
    }
}

mpcs = dict(
    axel = MPCSingleHomeDistributed(N, 'axel', params['axel'])
    # peak = MPCPeakStateDistributed(N, 'peak', params['peak'])
    )

dmpc = DistributedMPC(N, T, mpcs, -0.3)
dmpc.run_full()

#%%

mpc_axel = mpcs['axel']

pwr = mpc_axel.traj_full['P_hp']
figp, axp = plt.subplots()
axp.plot(pwr)
axp.legend()
axp.set_title('Power usage [W]')

room = mpc_axel.traj_full['room']
figr, axr = plt.subplots()
axr.plot(room)
axr.legend()
axr.set_title('Room temperature')

wall = mpc_axel.traj_full['wall']
figw, axw = plt.subplots()
axw.plot(wall)
axw.legend()
axw.set_title('Wall temperature')


traj_dv = dmpc.dual_variables_traj
x, y = np.meshgrid(np.arange(traj_dv.shape[1]), np.arange(traj_dv.shape[0]))
z = traj_dv
figdv, axdv = plt.subplots(subplot_kw={"projection": "3d"})
axdv.set_title('Dual variable values')
surf = axdv.plot_surface(x, y, z, cmap=cm.coolwarm)
axdv.zaxis.set_rotate_label(False)
axdv.set_xlabel('Time')
axdv.set_ylabel('Iteration')
axdv.set_zlabel('$\lambda$')

plt.show()
# %%
