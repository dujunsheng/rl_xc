from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

multi = []
single = []

no_stop = pd.read_csv('../record/no_stop_2_test_tripinfo.csv', sep=';')
stop = pd.read_csv('../conf/dqn_3_test_tripinfo.csv', sep=';')
data =pd.merge(no_stop[['tripinfo_id', 'tripinfo_duration']], stop[['tripinfo_id', 'tripinfo_duration']], on='tripinfo_id')

no_stop['speed_avg'] = no_stop['tripinfo_routeLength'].div(no_stop['tripinfo_duration'])
print("no_stop mean %.2f " % no_stop['speed_avg'].mean())

stop['speed_avg'] = stop['tripinfo_routeLength'].div(stop['tripinfo_duration'])
print("stop mean %.2f " % stop['speed_avg'].mean())

target = pd.read_csv('../conf/veh_log_stop.csv')
stop1 = stop[~stop['tripinfo_id'].isin(target['vehID'])]
stop1['speed_avg'] = stop1['tripinfo_routeLength'].div(stop1['tripinfo_duration'])
no_stop1 = no_stop[~no_stop['tripinfo_id'].isin(target['vehID'])]
no_stop1['speed_avg'] = no_stop1['tripinfo_routeLength'].div(no_stop1['tripinfo_duration'])
print("normal stop mean %.2f" % stop1['speed_avg'].mean())
print("normal no stop mean %.2f" % no_stop1['speed_avg'].mean())

stop2 = stop[stop['tripinfo_id'].isin(target['vehID'])]
stop2['speed_avg'] = stop2['tripinfo_routeLength'].div(stop2['tripinfo_duration'])
no_stop2 = no_stop[no_stop['tripinfo_id'].isin(target['vehID'])]
no_stop2['speed_avg'] = no_stop2['tripinfo_routeLength'].div(no_stop2['tripinfo_duration'])
print("normal stop mean %.2f" % stop2['speed_avg'].mean())
print("normal no stop mean %.2f" % no_stop2['speed_avg'].mean())

data2 =pd.merge(target[['vehID', 'in_time','out_time']],
               stop2[['tripinfo_id', 'tripinfo_stopTime']], left_on='vehID', right_on=['tripinfo_id'])
# data2 = data2[data2['tripinfo_stopTime']]
# data2['loss'] = data2['tripinfo_stopTime'] - (data2['out_time'] - data2['in_time'])

# 1.创建画布
plt.figure(figsize=(32, 8), dpi=100)

plt.subplots_adjust(bottom=0.15)

# 2.绘制折线图
# seaborn模块绘制分组的直方图和核密度图
#
# sns.distplot(data2['loss'], bins=int(100), kde=False, hist_kws={'color':'green'},
#              label=('不停车','直方图'), norm_hist=False)

# stop['avg_speed'] = (stop['tripinfo_routeLength'].div(stop['tripinfo_duration'])).round(2)
# sns.distplot(stop['avg_speed'], bins=int((stop['avg_speed'].max() - stop['avg_speed'].min())/0.1), kde=False, hist_kws={'color':'purple'},
#              label=('停车','直方图'), norm_hist=False)

# data.hist(grid=False,column="tripinfo_duration_y",bins=1000)

# sns.distplot(no_stop2['speed_avg'], bins = 100, kde = False, hist_kws = {'color':'green'},
#              label = ('不停车','直方图'),norm_hist=False)
#
# sns.distplot(stop2['speed_avg'], bins = 100, kde = False, hist_kws = {'color':'purple'},
#              label = ('停车','直方图'),norm_hist=False)

# data = data[~data['tripinfo_id'].isin(target['vehID'])]
# data_more_time = data['tripinfo_duration_y'] - data['tripinfo_duration_x']
# sns.distplot(data_more_time, bins=int(data_more_time.max() - data_more_time.min()), kde=False, hist_kws={'color':'green'},
#              label=('时间增加','直方图'), norm_hist=False)
# # 绘制女性年龄的核密度图
# sns.distplot(data['tripinfo_duration_y'], hist = False, kde_kws = {'color':'black', 'linestyle':'--'},
#              norm_hist = False, label = ('停车','核密度图'))
plt.xlabel('时间')
plt.ylabel('数量')
# plt.xlim([-25,100])
# plt.ylim([0,600])


# sns.set(style="whitegrid", font_scale=5.1)
plt.title('车辆行程时间', fontsize=16)
plt.legend()
# plt.savefig('111.pdf', bbox_inches='tight', pad_inches=0.5)
plt.show()
