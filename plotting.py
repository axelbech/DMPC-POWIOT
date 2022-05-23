import datetime
import os
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import numpy as np
from numpy import array
import pytz



# cpath = r'data\runs\MPCWrapper-N288T288-20220504-123908\\' # Old values
# dpath = r'data\runs\DMPCWrapper-N288T288-20220504-125642\\'
# dcpath = r'data\runs\MPCWrapper-N288T288-20220504-114723\\'

# cpath = r'data\runs\MPCWrapper-N288T288-20220506-233032\\' # Recent full run, no good hehe
# dpath = r'data\runs\DMPCWrapper-N288T288-20220507-000814\\'
# dcpath = r'data\runs\MPCWrapper-N288T288-20220506-233223\\'

# cpath = r'data\runs\MPCWrapperSerial-N288T288-linPeakCent\\' # Linear single peak
# dpath = r'data\runs\DMPCWrapperSerialProxGrad-N288T288-linPeakDist\\'
# dcpath = r'data\runs\MPCWrapperSerial-N288T288-linPeakDecent\\'

cpath = r'data\runs\MPCWrapperSerial-quadPeakCent2H\\' #Quadratic single peak
dpath = r'data\runs\DMPCWrapperSerial-quadPeakDist2H\\'
dcpath = r'data\runs\MPCWrapperSerial-quadPeakDecent2H\\'



ccpath = r'data\runs\MPCWrapperSerial-N288T288-quadPeakCent\MPCCentralizedSinglePeakConvex-cent.json' #quad cost
ddpath = r'data\runs\DMPCWrapperSerial-N288T288-quadPeakDist\MPCSinglePeakDistributed-peak.json'

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
        if name == 'peak' or name == 'peak_state' or name == 'dv_traj' or name == 'hourly_peak':
            continue
        res[name] = dict(
            traj_full = centralized_dict['traj_full'][name],
            params = centralized_dict['params'][name]
        )
    return res
    
def filename_splitter(fname):
    name = fname.split('-')[-1]
    name = name.split('/')[-1]
    # name = name.split('\\')[-1]
    name = name.split('.')[0]
    return name

def get_dv_padded(dv : list[list]):
    dv_len = len(dv[0])
    T = len(dv)
    dva = np.empty((T, dv_len+T-1))
    dva[:] = np.nan
    for t in range(T):
        dva[t, t:t+dv_len] = dv[t]
    return dva


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
    

def res_from_path(cpath, dpath, dcpath):
    cmpc = read_from_folder_centralized(cpath)
    dmpc = read_from_folder(dpath)
    dcmpc = read_from_folder(dcpath)
    l = list(dmpc.keys())
    for name in l:
        if not 'House' in name:
            del dmpc[name]
    return cmpc, dmpc, dcmpc

# temperature deviations sum of difference between ref temp and room temp, divided by 12
# Power consumption cost from multipying spot price w. P_hp use, summing all, divide by (12*100) for kr
# Total power consumption same just without multiplication, and divide by 12
def table_values(cpath, dpath, dcpath):
    cmpc, dmpc, dcmpc = res_from_path(cpath, dpath, dcpath)
    T = len(cmpc[list(cmpc.keys())[0]]['traj_full']['P_hp'])
    ref_temps = array([house['params']['opt_params']['reference_temperature'][:T] for house in cmpc.values()])
    spot_prices = array([house['params']['opt_params']['spot_price'][:T] for house in cmpc.values()])
    room_temps_c = array([house['traj_full']['room_temp'][:T] for house in cmpc.values()])
    room_temps_d = array([house['traj_full']['room_temp'][:T] for house in dmpc.values()])
    room_temps_dc = array([house['traj_full']['room_temp'][:T] for house in dcmpc.values()])
    P_hps_c = array([house['traj_full']['P_hp'][:T] for house in cmpc.values()])
    P_hps_d = array([house['traj_full']['P_hp'][:T] for house in dmpc.values()])
    P_hps_dc = array([house['traj_full']['P_hp'][:T] for house in dcmpc.values()])
    cpt, dpt, dcpt = get_total_power(cpath, dpath, dcpath)
    cph, dph, dcph = get_hourly_power(cpt, dpt, dcpt)
    cphmax = cph.max(); dphmax = dph.max(); dcphmax = dcph.max()
    print(f'Max Hourly Power Consumption [kWh]:\nCentralized: {cphmax}\nDistributed: {dphmax}\nDecentralized: {dcphmax}')
    temp_dev_c = np.sum(np.abs(room_temps_c-ref_temps))/12
    temp_dev_d = np.sum(np.abs(room_temps_d-ref_temps))/12
    temp_dev_dc = np.sum(np.abs(room_temps_dc-ref_temps))/12
    print(f'Temperature deviations [°Ch]:\nCentralized: {temp_dev_c}\nDistributed: {temp_dev_d}\nDecentralized: {temp_dev_dc}')
    pwr_cost_c = np.sum(P_hps_c * spot_prices)/(12*100)
    pwr_cost_d = np.sum(P_hps_d * spot_prices)/(12*100)
    pwr_cost_dc = np.sum(P_hps_dc * spot_prices)/(12*100)
    print(f'Heat Pump Power Cost [kr]:\nCentralized: {pwr_cost_c}\nDistributed: {pwr_cost_d}\nDecentralized: {pwr_cost_dc}')
    pwr_c = np.sum(P_hps_c)/(12)
    pwr_d = np.sum(P_hps_d)/(12)
    pwr_dc = np.sum(P_hps_dc)/(12)
    print(f'Heat Pump Power Consumption [kWh]:\nCentralized: {pwr_c}\nDistributed: {pwr_d}\nDecentralized: {pwr_dc}')

def retain_houses(res):
    if res.get('peak_state'):
        del res['peak_state']
    if res.get('peak'):
        del res['peak']
    if res.get('dv_traj'):
        del res['dv_traj']
    return res

def get_hourly_power(pc, pd, pdc):
    n = 12 # 12 5-minute steps in an hour
    hours = len(pc) // n
    pca = np.reshape(np.array(pc[:hours*n]), (hours, n))
    pda = np.reshape(np.array(pd[:hours*n]), (hours, n))
    pdca = np.reshape(np.array(pdc[:hours*n]), (hours, n))
    return np.average(pca,axis=1), np.average(pda,axis=1), np.average(pdca,axis=1)

def get_5m_time(length):
    start_time = datetime.datetime(2021, 11, 29, 0, 0, 0)#, tzinfo=pytz.timezone('Europe/Oslo'))
    time = [start_time + datetime.timedelta(seconds=i*300) for i in range(length)]
    return time

def get_1h_time(length):
    start_time = datetime.datetime(2021, 11, 29, 0, 0, 0)#, tzinfo=pytz.timezone('Europe/Oslo'))
    time = [start_time + datetime.timedelta(hours=i*1) for i in range(length)]
    return time
    
def plot_2_houses(cmpc_path, dmpc_path, dcmpc_path, title=''):
    cmpc = read_from_folder_centralized(cmpc_path)
    dmpc = read_from_folder(dmpc_path)
    if dmpc.get('DMPCCoordinator', False):
        del dmpc['DMPCCoordinator']
    dcmpc = read_from_folder(dcmpc_path)
    cmpc, dmpc, dcmpc = tuple(map(retain_houses, [cmpc, dmpc, dcmpc]))
    figr, axsr = plt.subplots(2, 1, figsize=(12,8))
    axsr[0].grid(); axsr[1].grid()
    figp, axsp = plt.subplots(1, figsize=(12,4))
    axsp.grid()
    axsr[0].set_ylabel('°C')
    axsp.set_ylabel('kW')
    pl = len(cmpc['House1']['traj_full']['P_hp'])
    pwrc = np.zeros(pl); pwrd = np.zeros(pl); pwrdc = np.zeros(pl)
    for i, house in enumerate(cmpc):
        roomc = cmpc[house]['traj_full']['room_temp']
        roomd = dmpc[house]['traj_full']['room_temp']
        roomdc = dcmpc[house]['traj_full']['room_temp']
        pwrc += cmpc[house]['traj_full']['P_hp']# + np.array(cmpc[house]['params']['opt_params']['ext_power_real'][:pl])
        pwrd += dmpc[house]['traj_full']['P_hp']# + np.array(dmpc[house]['params']['opt_params']['ext_power_real'][:pl])
        pwrdc += dcmpc[house]['traj_full']['P_hp']# + np.array(dcmpc[house]['params']['opt_params']['ext_power_real'][:pl])
        time = get_5m_time(len(roomc))
        axsr[i].plot(time, roomc, label='Centralized')
        axsr[i].plot(time, roomdc, label='Decentralized', alpha=0.7, linestyle='dashed')
        axsr[i].plot(time, roomd, label='Distributed', alpha=0.7,linestyle='dashed')
        axsr[i].set_title(house)
        axsr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axsp.plot(time, pwrc, label='Centralized')
    axsp.plot(time, pwrdc, label='Decentralized', alpha=0.7, linestyle='dashed')
    axsp.plot(time, pwrd, label='Distributed', alpha=0.7, linestyle='dashed')
    # axsp.set_title(house)
    axsp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    lines, labels = figr.axes[-1].get_legend_handles_labels()    
    figr.legend(lines, labels)
    lines, labels = figp.axes[-1].get_legend_handles_labels()    
    figp.legend(lines, labels)
    figr.suptitle(title+'Room Temperature', fontsize=16)
    figp.suptitle(title+'Total Heat Pump Power Use', fontsize=16)

    plt.show()
    
def plot_2_peak(cmpc_path, dmpc_path, title=''):
    with open(cmpc_path, 'r') as file:
        c = json.load(file)
    with open(dmpc_path, 'r') as file:
        d = json.load(file)
    cp = c['traj_full']['peak_state']
    dp = d['traj_full']['peak_state']
    time = get_5m_time(len(cp))
    fig, ax = plt.subplots(1, figsize=(12,4))
    plt.plot(time, cp, label='centralized')
    plt.plot(time, dp, label='distributed',alpha=0.7)
    ax.set_title(title+'Peak State', fontsize=16)
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.show()
    
def get_total_power(c_path, d_path, dc_path):
    c = read_from_folder_centralized(c_path)
    # c = retain_houses(c)
    d = read_from_folder(d_path)
    if d.get('DMPCCoordinator', False):
        del d['DMPCCoordinator']
    dc = read_from_folder(dc_path)
    c, d, dc = tuple(map(retain_houses, [c, d, dc]))
    cpt = np.zeros_like(np.array(c['House1']['traj_full']['P_hp']))
    dpt = np.zeros_like(cpt)
    dcpt = np.zeros_like(cpt)
    for house in c:        
        cpt += c[house]['traj_full']['P_hp']
        cpt += c[house]['params']['opt_params']['ext_power_real'][:len(c[house]['traj_full']['P_hp'])]
        dpt += d[house]['traj_full']['P_hp']
        dpt += d[house]['params']['opt_params']['ext_power_real'][:len(c[house]['traj_full']['P_hp'])]
        dcpt += dc[house]['traj_full']['P_hp']
        dcpt += dc[house]['params']['opt_params']['ext_power_real'][:len(c[house]['traj_full']['P_hp'])]
    return cpt, dpt, dcpt
    
def plot_power_hourly(c_path, d_path, dc_path, title=''):
    cpt, dpt, dcpt = get_total_power(c_path, d_path, dc_path)
    cp, dp, dcp = get_hourly_power(cpt, dpt, dcpt)
    time = get_1h_time(len(cp))
    plt.step(time, cp, label='centralized')
    plt.step(time, dcp, label='decentralized', alpha=0.7, linestyle='dashed')
    plt.step(time, dp, label='distributed', alpha=0.7, linestyle='dashed')
    plt.ylabel('kWh')
    f = plt.gcf()
    f.set_figheight(4)
    f.set_figwidth(12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.legend()
    plt.grid()
    plt.title(title+'Total Hourly Energy Consumption')
    plt.show()

def plot_power_total(c_path, d_path, dc_path, title=''):
    cpt, dpt, dcpt = get_total_power(c_path, d_path, dc_path)
    time = get_5m_time(len(cpt))
    f = plt.gcf()
    f.set_figheight(4)
    f.set_figwidth(12)
    plt.plot(time, cpt, label='centralized')
    plt.plot(time, dcpt, label='decentralized', alpha=0.7, linestyle='dashed')
    plt.plot(time, dpt, label='distributed',  alpha=0.7, linestyle='dashed')
    plt.ylabel('kW')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.legend()
    plt.grid()
    plt.title(title+'Total Power Consumption')
    plt.show()

title = 'Quadratic Peak Cost: '
# plot_2_houses(cpath, dpath, dcpath, title)
plot_power_hourly(cpath, dpath, dcpath, title)
# plot_power_total(cpath, dpath, dcpath, title)
# table_values(cpath, dpath, dcpath)
# plot_2_peak(ccpath, ddpath,title)