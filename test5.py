import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import json

def plot_dv_traj(dv_traj):
    if isinstance(dv_traj, list):
        dv_traj = np.array(dv_traj) 
    x, y = np.meshgrid(np.arange(dv_traj.shape[1]), np.arange(dv_traj.shape[0]))
    z = dv_traj
    figdv, axdv = plt.subplots(subplot_kw={"projection": "3d"})
    axdv.set_title('Dual variable values\nSliding window')
    surf = axdv.plot_surface(x, y, z, cmap=cm.coolwarm)
    axdv.zaxis.set_rotate_label(False)
    axdv.set_xlabel('Time')
    axdv.set_ylabel('Iteration')
    axdv.set_zlabel('$\lambda$')
    axdv.set_zlim(0,10)
    figdv.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
def get_dv_padded(dv : list[list]):
    dv_len = len(dv[0])
    T = len(dv)
    dva = np.empty((T, dv_len+T-1))
    dva[:] = np.nan
    for t in range(T):
        dva[t, t:t+dv_len] = dv[t]
    return dva

fpath = r'data\runs\MPCWrapperSerial-N288T288-20220513-152912\MPCCentralizedHomeSinglePeak-cent.json'

with open(fpath, 'r') as file:
    res = json.load(file)

dv = res['traj_full']['dv_traj']
dva = get_dv_padded(dv)
# print(np.max(np.array(dv)))
plot_dv_traj(dva)