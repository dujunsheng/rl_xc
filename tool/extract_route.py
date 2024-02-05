# 提取route

import pandas as pd

data = pd.read_csv('rou456.csv', sep=';')

data = pd.DataFrame(data['route_edges'])
data.dropna(how='all', inplace=True)

data.to_csv('all_route.csv', index=False, header=False)

