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

# d1 = r'data\housedata\MET_2021-11-29__1637167410-7f103004.pkl'
# d2 = r'data\housedata\MET_2021-11-30__1637167410-7f103004.pkl'

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
# temperature_full = temperature1 + temperature2

# outdoor_temp_5m = np.zeros(N+T)
# start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('Europe/Oslo'))
# for t in range(N+T):
#     date = start_time + datetime.timedelta(seconds=t*sampling_interval_seconds)
#     closest_idx = time_full.index(min(time_full, key=lambda d: abs(date - d)))
#     outdoor_temp_5m[t] = temperature_full[closest_idx]
    
# with open('data/housedata/outdoor_temp_5m.json', 'w') as file:
#     json.dump(list(outdoor_temp_5m), file)
    
    
with open(r'data\housedata\outdoor_temp_5m.json', 'r') as file:
    res = json.load(file)
    
start_time = datetime.datetime(2021, 11, 29, 0, 0, 0)
# otz = timezone('Europe/Oslo')
# start_time = otz.localize(start_time)
# start_time = datetime.datetime(2021, 11, 29, 0, 0, 0, tzinfo=timezone('UTC'))
l = len(res)
time = [0] * l
for i in range(l):
    time[i] = (start_time + datetime.timedelta(seconds=i*300))#.astimezone(timezone('Europe/Oslo'))
    
plt.plot(time, res)
ax = plt.gca()
fig = plt.gcf()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title('Outdoor Temperature')
ax.set_ylabel('°C')

plt.show()