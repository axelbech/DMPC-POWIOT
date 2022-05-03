import datetime
import json
import pickle
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytz import timezone

N = 288
T = 288
sampling_interval_seconds = 300

# d1 = r'data\housedata\tibber-realtime-home-up_2021-11-29__1637167410-7f101e8e.pkl'
# d2 = r'data\housedata\tibber-realtime-home-up_2021-11-30__1637167410-7f101e8e.pkl'

# d1 = r'data\power\tibber-realtime-home-up_2021-11-19__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-20__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-11-21__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-22__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-11-23__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-24__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-11-25__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-26__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-11-27__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-28__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-11-29__1637167410-7f101e8e.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-11-30__1637167410-7f101e8e.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-12-05__1638605720-52c2ece0.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-12-06__1638605720-52c2ece0.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-12-08__1638605720-52c2ece0.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-12-09__1638605720-52c2ece0.pkl'
# d1 = r'data\power\tibber-realtime-home-up_2021-12-10__1638605720-52c2ece0.pkl'
# d2 = r'data\power\tibber-realtime-home-up_2021-12-11__1638605720-52c2ece0.pkl'

# with open(d1, 'rb') as file:
#     res1 = pickle.load(file)
# with open(d2, 'rb') as file:
#     res2 = pickle.load(file)
    
# for key in res1:
#     value = res1[key]
#     exec(key + '1 = value')
# for key in res2:
#     value = res2[key]
#     exec(key + '2 = value')
    
# time_full = time1 + time2
# power_full = power1 + power2

# pwr_ext_5m = np.zeros(N+T)
# start_time = datetime.datetime(2021, 12, 10, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
# for t in range(N+T):
#     date = start_time + datetime.timedelta(seconds=t*sampling_interval_seconds)
#     closest_idx = time_full.index(min(time_full, key=lambda d: abs(date - d)))
#     pwr_ext_5m[t] = power_full[closest_idx] / 1000
    
# with open('data/power/pwr_ext_5m_1210.json', 'w') as file:
#     json.dump(list(pwr_ext_5m), file)



powers = []
filelist = os.listdir(r'data\power')
for file in filelist:
    if 'pwr_ext_5m_' in file:
        with open('data/power/' + file, 'r') as f:
            powers.append(json.load(f))
            
start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
l = len(powers[0])
time = [0] * l
for i in range(l):
    time[i] = start_time + datetime.timedelta(seconds=i*300)

powers = powers[:]
powerSingle = copy.copy(powers)
powers = []
for power in powerSingle:
    middleIdx = int(len(power) / 2)
    powers.append(power[:middleIdx])
    powers.append(power[middleIdx:])
time = time[:middleIdx]
fig, ax = plt.subplots()
for power in powers:
    ax.plot(time, power, linewidth=0.6)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title('Non-heatpump power use')
ax.set_ylabel('kW')

avg = np.array(powers)
avg = np.average(avg, axis=0)
kernel_size = 20
avg = np.pad(avg, (kernel_size, kernel_size), 'edge')
kernel = np.ones(kernel_size) / kernel_size
avg = np.convolve(avg, kernel, mode='same')
avg = avg[kernel_size:-kernel_size]
fig2, ax2 = plt.subplots()
ax2.plot(time, avg, linewidth=1)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax2.set_title('Smoothed, average, non-heatpump power use')
ax2.set_ylabel('kW')

avg_extended = np.concatenate((avg, avg)).tolist()
with open('data/power/pwr_ext_avg.json', 'w') as file:
    json.dump(avg_extended, file)

plt.show()


# with open(r'data\spotdata\spot_price_5m.json', 'r') as file:
#     res = json.load(file)
    
# start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
# l = len(res)
# time = [0] * l
# for i in range(l):
#     time[i] = start_time + datetime.timedelta(seconds=i*300)
    
# plt.plot(time, res)
# ax = plt.gca()
# fig = plt.gcf()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax.set_title('Spot Prices')
# ax.set_ylabel('øre/kWh' )

# plt.show()
