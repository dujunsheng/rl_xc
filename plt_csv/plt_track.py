from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import plot_trajectories as ptt
from sumolib.options import ArgumentParser  # noqa


# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# font1 = font_manager.FontProperties(fname='/Users/juns/Downloads/FangZhengShuSongJianTi/FangZhengShuSongJianTi-1.ttf',
#                                     size=30)

multi = []
single = []

no_stop = pd.read_csv('../record/no_stop_test_info_more.csv', sep=';')
stop = pd.read_csv('../record/stop_test_info_more.csv', sep=';')
data =pd.merge(no_stop[['tripinfo_id', 'tripinfo_duration']], stop[['tripinfo_id', 'tripinfo_duration']], on='tripinfo_id')
target = pd.read_csv('../conf/veh_log_test.csv')

data = data[~data['tripinfo_id'].isin(target['vehID'])][data['tripinfo_duration_y'] > data['tripinfo_duration_x'] + 200]
# target_ids = data[data['tripinfo_duration_y'] < data['tripinfo_duration_x'] + 1000]['tripinfo_id'].values
target_ids = data['tripinfo_id'].values

print(target_ids)

# options = ptt.getOptions("-t td -o test1.png ../record/no_stop_test_info_more_fcd.xml")
# options.__setattr__('filterIDs', set(target_ids))
# ptt.main(options)


options = ptt.getOptions("-t tg -o test3.png ../record/stop_test_info_more_fcd.xml")
options.__setattr__('filterIDs', set(['221888670#10__01.37']))
ptt.main(options)



