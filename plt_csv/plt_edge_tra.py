import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 读取停车指令发送时刻
# target = pd.read_csv('../sim_dict.csv')
target2 = pd.read_csv('../conf/veh_stop_truth.csv')
# data = pd.read_csv('../conf/random_stop_test_fcd4calib.csv', sep=';')

stop = pd.read_csv('../conf/test_tripinfo_备份.csv', sep=';')
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
# 1.创建画布
plt.figure(figsize=(32, 8), dpi=100)

plt.subplots_adjust(bottom=0.15)
sns.distplot(stop_truth['tripinfo_duration'], hist = {'color':'green'}, kde_kws = False,
             norm_hist = False, label = ('停车','核密度图'))

plt.xlabel('时间')
plt.ylabel('数量')
# plt.xlim([-25,100])
# plt.ylim([0,600])


# sns.set(style="whitegrid", font_scale=5.1)
plt.title('车辆行程时间', fontsize=16)
# plt.legend()
# plt.savefig('111.pdf', bbox_inches='tight', pad_inches=0.5)
plt.show()