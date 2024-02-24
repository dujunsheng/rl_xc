import pandas as pd
import math

# 读取停车指令发送时刻
target = pd.read_csv('../sim_dict.csv')
target2 = pd.read_csv('../conf/veh_log_stop.csv')
# data = pd.read_csv('../conf/test_fcd4calib.csv', sep=';')

stop = pd.read_csv('../conf/test_tripinfo.csv', sep=';')
stop2 = stop[stop['tripinfo_id'].isin(target['vehid']) & (stop['tripinfo_stopTime'] == 0)]

data2 =pd.merge(target2[['vehID', 'in_time','out_time']],
               stop[['tripinfo_id', 'tripinfo_duration']], left_on='vehID', right_on=['tripinfo_id'])

data2['valid_duration'] = data2['out_time'] - data2['in_time']

data2['e2'] = (data2['valid_duration'] - data2['tripinfo_duration']) / data2['valid_duration']

# 行程时间平均误差
print(data2['e2'].abs().mean())


normal = pd.read_csv('../record/no_stop_2_test_tripinfo.csv', sep=';')
data3 = pd.merge(data2[['vehID', 'valid_duration']],
               normal[['tripinfo_id', 'tripinfo_duration']], left_on='vehID', right_on=['tripinfo_id'])

data3['valid_stopTime'] = data3['valid_duration'] - data3['tripinfo_duration']


data3 = pd.merge(data3[['vehID', 'valid_stopTime']],
               stop[['tripinfo_id', 'tripinfo_stopTime']], left_on='vehID', right_on=['tripinfo_id'])

data3['e3'] = (data3['valid_stopTime'] - data3['tripinfo_stopTime']) / data3['valid_stopTime']

#停车时间平均误差
print(data3['e3'].abs().mean())

cnt_e1 = 0
cnt_e2 = 0
for v in stop2['tripinfo_id'].values:
    lanes = data[(data['vehicle_id'] == v) & (data['timestep_time'] > target[(target['vehid'] == v)]['depart'].values[0])]['vehicle_lane'].values
    for lane in lanes:
        if lane.split('_')[1] == '0':
            print()


print()