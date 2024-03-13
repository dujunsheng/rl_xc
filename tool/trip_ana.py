# 分析车辆数据，计算目标车辆

# 获取经过 ['64422', '63366'] 道路车辆

# 提取route
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.figure(figsize=(8, 6))


data1 = pd.read_csv('./rou789.csv', sep=';')
data2 = pd.read_csv('./hello.rou.csv', sep=';')
data3 = pd.read_csv('./added_veh_routes.csv', sep=';')

data4 = pd.concat([data1[['vehicle_id', 'route_edges']], data2[['vehicle_id', 'route_edges']], data3[['vehicle_id', 'route_edges']]])
data4.dropna(how='all', inplace=True)

tars_road = {'64422', '63366'}
tars_veh = set()

for ind, val in data4.iterrows():
    if tars_road.issubset(set(val['route_edges'].split(' '))):
        tars_veh.add(val['vehicle_id'])

data5 = pd.DataFrame(columns=['vehID'], data=tars_veh)
data6 = pd.read_csv('../conf/dqn_3_test_tripinfo.csv', sep=';')
data10 = pd.read_csv('../conf/no_stop_tripinfo.csv', sep=';')
data20 = pd.read_csv('../conf/dqn_tripinfo.csv', sep=';')

data7 = data6[data6['tripinfo_id'].isin(data5['vehID'])]
data11 = data10[data10['tripinfo_id'].isin(data5['vehID'])]
data21 = data20[data20['tripinfo_id'].isin(data5['vehID'])]


data7['avg_speed'] = (data7['tripinfo_routeLength'].div(data7['tripinfo_duration'])).round(2)
# sns.distplot(data7['avg_speed'], bins=int((data7['avg_speed'].max() - data7['avg_speed'].min())/0.1),
#              # kde_kws={"color": 'blue', "lw": 3 },
#              kde=False,
#              hist_kws={'color': 'purple'},
#              label=('全部车辆','直方图'), norm_hist=False)
print(tars_veh)

# plt.xlim([-25,100])
# plt.ylim([0,600])

tar = pd.read_csv('../conf/veh_stop_truth.csv')
#
# data8 = data7[data7['tripinfo_id'].isin(tar['vehID'])]
data8 = data7
data8['avg_speed'] = (data8['tripinfo_routeLength'].div(data8['tripinfo_duration'])).round(2)
# (data8['tripinfo_duration'].max() - data8['tripinfo_duration'].min()
sns.distplot(data8['tripinfo_duration'], bins=300,
             # kde_kws={"color": 'green', "lw": 3 },
             kde=False,
             hist_kws={'color':'green'},
             label=('基准值','直方图'), norm_hist=False)


sns.distplot(data11['tripinfo_duration'],
             # kde_kws={"color": 'green', "lw": 3 },
             kde=False,
             hist_kws={'color':'purple'},
             label=('单一校准值流程结果','直方图'), norm_hist=False)

sns.distplot(data21['tripinfo_duration'],
             # kde_kws={"color": 'green', "lw": 3 },
             kde=False,
             hist_kws={'color':'blue'},
             label=('结合停车行为的校准结果','直方图'), norm_hist=False)

# plt.xlim([-25,100])
# plt.ylim([0,600])

plt.xlabel('个体行程时间')
plt.ylabel('数量')


# sns.set(style="whitegrid", font_scale=5.1)
plt.title('行程时间分布', fontsize=20)
plt.legend()
plt.savefig('222.pdf', bbox_inches='tight', pad_inches=0.5)
plt.show()