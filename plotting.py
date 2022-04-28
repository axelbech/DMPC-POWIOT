import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import os
import json

# folder_path = 'data/runs/DistributedMPC-N50T50-20220425-203532/'
folder_path = 'data/runs/DistributedMPC-N50T50-20220426-111014/'
fdlin = 'data/runs/DistributedMPC-N100T100-20220427-111643/'
fdquad = 'data/runs/DistributedMPC-N100T100-20220427-111743/'
fclin = 'data/runs/MPCsWrapper-N100T100-20220427-111800/'
fcquad = 'data/runs/MPCsWrapper-N100T100-20220427-111817/'

def read_from_folder(folder_path):
    res = {}
    for fname in os.listdir(folder_path):
        name = filename_splitter(fname)
        with open(folder_path + fname, 'r') as file:
            res[name] = json.load(file)
    return res

def read_from_folder_centralized(folder_path):
    res = {}
    for fname in os.listdir(folder_path):
        # name = filename_splitter(fname)
        with open(folder_path + fname, 'r') as file:
            centralized_dict = json.load(file)
    names = list(centralized_dict['traj_full'].keys())
    for name in names:
        res[name] = dict(
            traj_full = centralized_dict['traj_full'][name]
        )
    return res
    
def filename_splitter(fname):
    name = fname.split('-')[-1]
    name = name.split('/')[-1]
    # name = name.split('\\')[-1]
    name = name.split('.')[0]
    return name


def plot_dv_traj(dv_traj):
    if isinstance(dv_traj, list):
        dv_traj = np.array(data['dv_traj']) 
    x, y = np.meshgrid(np.arange(dv_traj.shape[1]), np.arange(dv_traj.shape[0]))
    z = dv_traj
    figdv, axdv = plt.subplots(subplot_kw={"projection": "3d"})
    axdv.set_title('Dual variable values\nSliding window')
    surf = axdv.plot_surface(x, y, z, cmap=cm.coolwarm)
    axdv.zaxis.set_rotate_label(False)
    axdv.set_xlabel('Time')
    axdv.set_ylabel('Iteration')
    axdv.set_zlabel('$\lambda$')
    figdv.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
def plot_house_temperatures(houses):
    fig, axs = plt.subplots(1, 2)
    for house in houses:
        traj_full = houses[house]['traj_full']
        room_temp = traj_full['room_temp']
        wall_temp = traj_full['wall_temp']
        axs[0].plot(room_temp, label=house)
        axs[1].plot(wall_temp, label=house)
    axs[0].legend()
    axs[1].legend()
    plt.show()

# data = read_from_folder(fdquad)
data = read_from_folder_centralized(fclin)
del data['peak_state']
plot_house_temperatures(data)

# house_data = dict(data)
# del house_data['dv_traj']
# del house_data['peak']
# plot_house_temperatures(dict(house_data))

    
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


