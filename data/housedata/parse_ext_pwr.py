import datetime
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytz import timezone

N = 288
T = 288
sampling_interval_seconds = 300

d1 = r'data\housedata\tibber-realtime-home-up_2021-11-29__1637167410-7f101e8e.pkl'
d2 = r'data\housedata\tibber-realtime-home-up_2021-11-30__1637167410-7f101e8e.pkl'

with open(d1, 'rb') as file:
    res1 = pickle.load(file)
with open(d2, 'rb') as file:
    res2 = pickle.load(file)
    
for key in res1:
    value = res1[key]
    exec(key + '1 = value')
for key in res2:
    value = res2[key]
    exec(key + '2 = value')
    
time_full = time1 + time2
power_full = power1 + power2

pwr_ext_5m = np.zeros(N+T)
start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
for t in range(N+T):
    date = start_time + datetime.timedelta(seconds=t*sampling_interval_seconds)
    closest_idx = time_full.index(min(time_full, key=lambda d: abs(date - d)))
    pwr_ext_5m[t] = power_full[closest_idx] / 1000
    
with open('data/housedata/pwr_ext_5m.json', 'w') as file:
    json.dump(list(pwr_ext_5m), file)


