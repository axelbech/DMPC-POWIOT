import datetime
import os
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import numpy as np
import pytz



# cpath = r'data\runs\MPCWrapper-N288T288-20220504-123908\\' # Old values
# dpath = r'data\runs\DMPCWrapper-N288T288-20220504-125642\\'
# dcpath = r'data\runs\MPCWrapper-N288T288-20220504-114723\\'

# cpath = r'data\runs\MPCWrapper-N288T288-20220506-233032\\' # Recent full run, no good hehe
# dpath = r'data\runs\DMPCWrapper-N288T288-20220507-000814\\'
# dcpath = r'data\runs\MPCWrapper-N288T288-20220506-233223\\'


# cpath = r'data\runs\MPCWrapper-N100T10-20220507-103535\\'
# dpath = r'data\runs\DMPCWrapperSerial-N100T10-20220507-103521\\'
# dcpath = r'data\runs\MPCWrapper-N100T10-20220507-103528\\'

ddpath = r'data\runs\DMPCWrapper-N25T25-20220505-092941\MPCPeakStateDistributedQuadratic-peak.json'
ccpath = r'data\runs\MPCWrapper-N25T25-20220505-092634\MPCCentralizedHomePeakQuadratic-cent_quad.json'


def read_from_folder(folder_path):
    res = {}
    for fname in os.listdir(folder_path):
        if fname == 'run_time_seconds.json':
            continue
        name = filename_splitter(fname)
        with open(folder_path + fname, 'r') as file:
            res[name] = json.load(file)
    return res

def read_from_folder_centralized(folder_path):
    res = {}
    for fname in os.listdir(folder_path):
        if fname == 'run_time_seconds.json':
            continue
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
    
def remove_peak(res):
    if res.get('peak_state'):
        del res['peak_state']
    if res.get('peak'):
        del res['peak']
    return res

def get_5m_time(length):
    start_time = datetime.datetime(2021, 11, 29, 0, 0, 0)#, tzinfo=pytz.timezone('Europe/Oslo'))
    time = [start_time + datetime.timedelta(seconds=i*300) for i in range(length)]
    return time
    
def plot_2_houses(cmpc_path, dmpc_path, dcmpc_path):
    cmpc = read_from_folder_centralized(cmpc_path)
    dmpc = read_from_folder(dmpc_path)
    if dmpc.get('DMPCCoordinator', False):
        del dmpc['DMPCCoordinator']
    dcmpc = read_from_folder(dcmpc_path)
    cmpc, dmpc, dcmpc = tuple(map(remove_peak, [cmpc, dmpc, dcmpc]))
    figr, axsr = plt.subplots(1, 2)
    figp, axsp = plt.subplots(1, 2)
    axsr[0].set_ylabel('°C')
    axsp[0].set_ylabel('kW')
    for i, house in enumerate(cmpc):
        roomc = cmpc[house]['traj_full']['room_temp']
        roomd = dmpc[house]['traj_full']['room_temp']
        roomdc = dcmpc[house]['traj_full']['room_temp']
        pwrc = cmpc[house]['traj_full']['P_hp']
        pwrd = dmpc[house]['traj_full']['P_hp']
        pwrdc = dcmpc[house]['traj_full']['P_hp']
        time = get_5m_time(len(roomc))
        axsr[i].plot(time, roomc, label='Centralized')
        axsr[i].plot(time, roomd, label='Distributed')
        axsr[i].plot(time, roomdc, label='Decentralized')
        axsr[i].set_title(house)
        axsp[i].plot(time, pwrc, label='Centralized')
        axsp[i].plot(time, pwrd, label='Distributed')
        axsp[i].plot(time, pwrdc, label='Decentralized')
        axsp[i].set_title(house)
        axsr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axsp[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    lines, labels = figr.axes[-1].get_legend_handles_labels()    
    figr.legend(lines, labels)
    lines, labels = figp.axes[-1].get_legend_handles_labels()    
    figp.legend(lines, labels)
    figr.suptitle('Room Temperature', fontsize=16)
    figp.suptitle('Heat Pump Power Use', fontsize=16)
    plt.show()
    
def plot_2_peak(cmpc_path, dmpc_path):
    with open(cmpc_path, 'r') as file:
        c = json.load(file)
    with open(dmpc_path, 'r') as file:
        d = json.load(file)
    cp = c['traj_full']['peak_state']
    dp = d['traj_full']['peak_state']
    time = get_5m_time(len(cp))
    plt.plot(time, cp, label='centralized')
    plt.plot(time, dp, label='distributed')
    ax = plt.gca()
    ax.set_title('Peak state')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.show()
    
plot_2_houses(cpath, dpath, dcpath)
# plot_2_peak(ccpath, ddpath)





    
# plt.plot(time, res)
# ax = plt.gca()
# fig = plt.gcf()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax.set_title('Spot Prices')
# ax.set_ylabel('øre/kWh' )

# plt.show()

# house_data = dict(data)
# del house_data['dv_traj']
# del house_data['peak']
# plot_house_temperatures(dict(house_data))

    
# peak_state = peak['traj_full']['peak']
# fig,ax = plt.subplots()
# plt.plot(peak_state)

# P_hp = house['traj_full']['P_hp']
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


