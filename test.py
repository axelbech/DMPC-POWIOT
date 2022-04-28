import pickle
from datetime import datetime


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytz import timezone

filename = "data/spotdata/SpotData2021.pkl"

with open(filename, 'rb') as file:
    res = pickle.load(file)
    
dt = datetime(2021, 11, 29, 0, 0, tzinfo=timezone('UTC'))

start = res['Time_start']
end = res['Time_end']
price = res['Price']

start = [d.astimezone(timezone('Europe/Oslo')) for d in start]

i = start.index(dt)

plt.plot(start[i:i+24], price[i:i+24])
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.show()

