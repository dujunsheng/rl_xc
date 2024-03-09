# 统计真实的停车时间
# 统计真实的停车时刻

import pandas as pd

data = pd.read_csv('../conf/dqn_3_test_tripinfo.csv', sep=';')
target = pd.read_csv('../conf/veh_log_stop.csv')

stop = data[data['tripinfo_id'].isin(target['vehID'])][['tripinfo_id', 'tripinfo_depart', 'tripinfo_arrival', 'tripinfo_stopTime']]
stoptocsv = stop[['tripinfo_id', 'tripinfo_depart', 'tripinfo_arrival']]
stoptocsv.to_csv('veh_stop_truth.csv', index=False)

fcd = pd.read_csv('../conf/dqn_3_test_fcd4calib.csv', sep=';')
sim = fcd[fcd['vehicle_id'].isin(target['vehID'])]
stop_time = {}
for i in target['vehID'].values:
    veh_spd = fcd[fcd['vehicle_id'] == i][['timestep_time', 'vehicle_id', 'vehicle_speed']]
    cnt = 0
    for _, i_v in veh_spd.iterrows():
        if i_v['vehicle_speed'] == 0:
            cnt += 1
        else:
            if data[data['tripinfo_id'] == i].empty:
                print(i)
                continue
            elif cnt == stop[stop['tripinfo_id'] == i]['tripinfo_stopTime'].values[0]:
                stop_time[i] = i_v['timestep_time'] - cnt
                continue
            cnt = 0

print(stop_time)

df = pd.DataFrame.from_dict(stop_time, orient='index',columns=['depart'])
df = df.reset_index().rename(columns = {'index':'vehid'})
df.to_csv('sim_dict_truth.csv', index=False)
