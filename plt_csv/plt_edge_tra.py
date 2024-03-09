import pandas as pd
import math

# 读取停车指令发送时刻
# target = pd.read_csv('../sim_dict.csv')
target2 = pd.read_csv('../conf/veh_stop_truth.csv')
# data = pd.read_csv('../conf/random_stop_test_fcd4calib.csv', sep=';')

stop = pd.read_csv('../conf/test_tripinfo.csv', sep=';')
stop2 = stop[stop['tripinfo_id'].isin(target2['vehID']) & (stop['tripinfo_stopTime'] == 0)]

data2 =pd.merge(target2[['vehID', 'in_time','out_time']],
               stop[['tripinfo_id', 'tripinfo_duration']], left_on='vehID', right_on=['tripinfo_id'])

data2['valid_duration'] = data2['out_time'] - data2['in_time']

data2['e2'] = (data2['valid_duration'] - data2['tripinfo_duration']) / data2['valid_duration']

# 行程时间平均误差
print(data2['e2'].abs().mean())


stop_truth = pd.read_csv('../conf/dqn_3_test_tripinfo.csv', sep=';')
stop_truth = stop_truth[stop_truth['tripinfo_id'].isin(target2['vehID'])]

data3 = pd.merge(stop_truth[['tripinfo_id', 'tripinfo_stopTime']],
               stop[['tripinfo_id', 'tripinfo_stopTime']], left_on='tripinfo_id', right_on=['tripinfo_id'])

data3['e3'] = (data3['tripinfo_stopTime_x'] - data3['tripinfo_stopTime_y']) / data3['tripinfo_stopTime_x']

#停车时间平均误差
print(data3['e3'].apply(abs).mean())

# cnt_e1 = 0
# cnt_e2 = 0
# for v in stop2['tripinfo_id'].values:
#     lanes = data[(data['vehicle_id'] == v) & (data['timestep_time'] > target[(target['vehid'] == v)]['depart'].values[0])]['vehicle_lane'].values
#     for lane in lanes:
#         if lane.split('_')[1] == '0':
#             print()


print()