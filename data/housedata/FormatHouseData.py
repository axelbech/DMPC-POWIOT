import os
import sys
import pickle
from datetime import date, timedelta
import datetime
import numpy as np


BackupPath = '/Users/sebasgro/POWIOT/Data/HouseTrondheim'

#File types
Types = {'TibberUp'   : 'tibber-realtime-home-up_',
         'TibberPump' : 'tibber-realtime-home-pumps_',
         'Tibber'     : 'tibber_',
         'Sensibo'    : 'sensibo_',
         'MET'        : 'MET_'}
                  
Files = os.listdir(BackupPath)


def Merge_Data(dict_source,dict_addon):
    # Iterate over all values of given dictionary
    for key in dict_addon.keys():
        # Check if value is of dict type
        if isinstance(dict_addon[key], dict):
            # If value is dict then iterate over all its values
            Merge_Data(dict_source[key],dict_addon[key])
        else:
            # If value is not dict type then yield the value
            if key in dict_source:
                dict_source[key] += dict_addon[key]
            else:
                dict_source[key]  = dict_addon[key]
    return dict_source

def ReOrder(dic,index):
    for key in dic.keys():
        if isinstance(dic[key],dict):
            print(key)
            dic[key] = ReOrder(dic[key],index)
        else:
            print('Reorder : '+key)
            list = []
            for index_i in index:
                list.append(dic[key][index_i])
            dic[key] = list

    return dic



def SortByTime(dict):
    print('Sort data by time')
    if 'time' in dict.keys():
        print('---------------------------')
        index = np.argsort(dict['time'])
        dict = ReOrder(dict,index)
    else:
        for key in dict.keys():
            print(key)
            dict[key] = SortByTime(dict[key])
    return dict
        
def CreateAbsTime(dict,Start):
    print('Build absolute time')
    if 'time' in dict.keys():
        # Create absolute time
        dict['time_abs'] = []
        for time in dict['time']:
            dict['time_abs'].append( (time-Start).total_seconds()/3600. )
    else:
        for key in dict.keys():
            dict[key] = CreateAbsTime(dict[key],Start)
    return dict

"""
#Check MET
print('datetime object in temperature list in files:')
METData = {}
for file in Files:
    file_ext = os..splitext(file)[1]
path
    if file_ext=='.pkl' and 'MET' in file:
        f = open(BackupPath+'/'+file,"rb")
        Data_f = pickle.load(f)
        f.close()
        METData = Merge_Data(METData,Data_f)
        
        datetimeinstance = False
        for item in Data_f['temperature']:
            if isinstance(item, datetime.datetime) and not(datetimeinstance):
                print(file)
                datetimeinstance = True
                sys.exit()
sys.exit()
"""



# Extract data:
Data = {}
for file in Files:
    file_ext = os.path.splitext(file)[1]
    if file_ext=='.pkl':
        print('File : '+file)
        f = open(BackupPath+'/'+file,"rb")
        try:
            Data_f = pickle.load(f)
            ValidFile = True
        except:
            print('File: '+file+' is empty')
            ValidFile = False
        f.close()
        
        if 'MET' in file:
            #Check for invalid temperature lists
            for item in Data_f['temperature']:
                if isinstance(item, datetime.datetime):
                    ValidFile = False
        
        if ValidFile:
            for type in Types:
                if Types[type] in file:
                    if type in Data:
                        #Data of this type already exist, merge
                        Data[type] = Merge_Data(Data[type],Data_f)
                    else:
                        Data[type] = Data_f

# Sort data by time:
SortByTime(Data)

# Create absolute time
Start = Data['MET']['time'][0]
CreateAbsTime(Data,Start)

for type in Types:
    Data[type]['time_start'] = Start

#Save all data together
print('Save data in one file')
f = open('AllData.pkl',"wb")
pickle.dump(Data,f, protocol=2)
f.close()

#Save data by type
print('Save data type by type:')
for type in Types:
    print('Saving '+type)
    f = open('Data_'+type+'.pkl',"wb")
    pickle.dump(Data[type],f, protocol=2)
    f.close()


