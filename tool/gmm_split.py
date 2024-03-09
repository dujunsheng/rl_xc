from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np


# GMM
#Means: [29.4, 10.0, 38.9], Standard Deviations: [4.6, 3.1, 7.9]

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
data7 = data6[data6['tripinfo_id'].isin(data5['vehID'])]
data7['avg_speed'] = (data7['tripinfo_routeLength'].div(data7['tripinfo_duration'])).round(2)

tar = pd.read_csv('../conf/veh_stop_truth.csv')

data8 = data7[data7['tripinfo_id'].isin(tar['vehID'])]

gmm = GaussianMixture(n_components=2, weights_init=[0.1028, 0.8972])
gmm.fit(data7['avg_speed'].to_numpy().reshape(-1, 1))

means = gmm.means_

# Conver covariance into Standard Deviation
standard_deviations = gmm.covariances_**0.5

# Useful when plotting the distributions later
weights = gmm.weights_


print(f"Means: {means}, Standard Deviations: {standard_deviations}, Weight: {weights}")

err = abs(data7['avg_speed'].mean() - (means[0] * weights[0] + means[1] * weights[1]))

print(f"err: {err}")

# 停车时间 = 行程时间 - 交通通畅时所需时间

# 按照分布采样
np.random.choice(data7['avg_speed'].to_numpy())

