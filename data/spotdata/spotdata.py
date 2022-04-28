import pickle
import datetime

import numpy as np

with open('SpotData2021.pkl', 'rb') as file:
    res = pickle.load(file)

time_interval = 300 # seconds in 5 minutes
hourly_slots = int(3600 / time_interval) # Assumes this divides evenly
start_time = datetime.datetime(2021, 11, 29, 0, tzinfo=datetime.timezone.utc)
end_time = datetime.datetime(2021, 12, 1, 0, tzinfo=datetime.timezone.utc)

start_idx = res['Time_start'].index(start_time)
end_idx = res['Time_start'].index(end_time)

prices = np.zeros((hourly_slots * (end_idx - start_idx),))

for idx in range(start_idx, end_idx):
    full_idx = (idx - start_idx) * hourly_slots
    price = res['Price'][idx]
    prices[full_idx : full_idx + hourly_slots] = price

