#%%
from mpc import MPCSingleHomeDistributed, MPCPeakStateDistributed
from dmpc import DistributedMPC

import numpy as np
import matplotlib.pyplot as plt

N = 10
T = 10

n_houses = 1
p_max = 1.5

outdoor_temperature = [10 for _ in range(N+T)]
reference_temperature = [23 for _ in range(N+T)]
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
            'room': 15,
            'wall': 10,
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

mpcs = dict(
    axel = MPCSingleHomeDistributed(N, 'axel', params['axel'])
    # peak = MPCPeakStateDistributed(N, 'peak', params['peak'])
    )

dmpc = DistributedMPC(N, T, mpcs, 0)
dmpc.run_full()

#%%

mpc_axel = mpcs['axel']
pwr = mpc_axel.traj_full['P_hp']
fig, ax = plt.subplots()
ax.plot(pwr)
ax.legend()
ax.set_title('Power usage [W]')