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

filename = r'data\spotdata\SpotData2021.pkl'

with open(filename, 'rb') as file:
    res = pickle.load(file)
    
for key in res:
    value = res[key]
    exec(key + ' = value')

time_full = Time_start



spot_price_5m = np.zeros(N+T)
start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
for t in range(N+T):
    date = start_time + datetime.timedelta(seconds=t*sampling_interval_seconds)
    closest_idx = time_full.index(min(time_full, key=lambda d: abs(date - d)))
    spot_price_5m[t] = Price[closest_idx]
    
# with open('data/spotdata/spot_price_5m.json', 'w') as file:
#     json.dump(list(spot_price_5m), file)