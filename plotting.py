import matplotlib.pyplot as plt
from matplotlib import cm

import os
import json

folder_path = 'data/runs/DistributedMPC-N50T50-20220425-203532/'

def read_from_folder(folder_path):
    res = {}
    for fname in os.listdir(folder_path):
        name = filename_splitter(fname)
        with open(fname, 'r') as file:
            res[name] = json.load(file)
    return res
    
def filename_splitter(fname):
    name = fname.split('-')[-1]
    name = name.split('/')[-1]
    name = name.split('\\')[-1]
    return name

data = read_from_folder(folder_path)

    
# peak_state = peak['traj_full']['peak']
# fig,ax = plt.subplots()
# plt.plot(peak_state)

# P_hp = home['traj_full']['P_hp']
# fig,ax = plt.subplots()
# ax.plot(P_hp)

# plt.show()
    

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