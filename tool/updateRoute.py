import xml.dom.minidom
import random

from xml.etree.ElementTree import fromstring, Element
from matplotlib import pyplot as plt
import numpy as np
from fitter import Fitter
import uuid
import pandas as pd

# 产生更多停车数据
route_file1 = 'rou456.xml'
route_file2 = 'added_veh_routes2.xml'
route_file3 = '../conf/hello.rou.xml'
route_file4 = '../conf/rou456.xml'

add_cnt = 15
ori_cnt = 32


def create_file():
    domTree = xml.etree.ElementTree.parse(route_file2)
    root = domTree.getroot()

    '设置rou.xml文件'

    add_vehicle_edge = {}
    routes = []

    inputs = root.findall("vehicle")
    for item in inputs:
        for rou in item:
            add_vehicle_edge[item.get('id')] = (item.get('depart'), rou.get('edges'))
            routes.append(rou.get('edges'))
        for i in range(add_cnt):
            root.append(item)

    domTree.write(route_file3)


def update_file():
    domTree = xml.etree.ElementTree.parse(route_file3)
    root = domTree.getroot()
    cnt = 1

    inputs = root.findall("vehicle")
    for item in inputs[ori_cnt:]:
        item.set('id', 'added_' + str(cnt + ori_cnt + 1))
        # 随机获取发车时间
        item.set('depart', str(round(random.random(), 2) * (30600 - 25201) + 25201))
        # item.set('color', '1,0,0')
        cnt += 1
        # print(item.get('id'))
    # for item in inputs:
    #     item.set('vtype', 'vtype' + 'added_' + item.get('id'))
    domTree.write(route_file3)


def sort_vehicle():
    domTree = xml.etree.ElementTree.parse(route_file3)
    root = domTree.getroot()

    inputs = root.findall("vehicle")
    _inputs = sorted(inputs, key=lambda item: item.get('depart'))

    for item in inputs:
        root.remove(item)
    root.extend(_inputs)
    inputs = []
    # for item in _inputs:
    #     ele = Element('vType',{
    #         'accel':'3'
    #     })
    #
    #     ele.set('accel', '3')
    #     ele.set('color', '1,0,0')
    #     ele.set('decel', '4')
    #
    #     uuid_obj = uuid.uuid4()
    #     uuid_str = uuid_obj.hex
    #     ele.set('hphm', str(uuid_str))
    #     ele.set('hpzl', '01' if random.random() < 0.03 else "02")
    #     ele.set('id', item.get('vtype'))
    #     ele.set('lcCooperative', '1')
    #     ele.set('lcKeepRight', '1')
    #     ele.set('lcSpeedGain', '1')
    #     ele.set('lcStrategic', '1')
    #     ele.set('length', '4.3')
    #     b = np.random.normal(loc=9.077809379, scale=3.265970068, size=1)[0]
    #     while b < 5 or b > 16:
    #         b = np.random.normal(loc=9.077809379, scale=3.265970068, size=1)[0]
    #     ele.set('maxSpeed', str(b))
    #     ele.set('minGap', '1')
    #     ele.set('sigma', '0.0')
    #
    #     a = np.random.normal(loc=0.60030, scale=0.001848, size=1)[0]
    #     while a < 0.5 or a > 0.76:
    #         a = np.random.normal(loc=0.60030, scale=0.001848, size=1)[0]
    #
    #     ele.set('tau', str(a))
    #     ele.set('width', '1.7')
    #
    #     inputs.append(ele)
    #     inputs.append(item)
    # root.extend(inputs)
    domTree.write(route_file3)


def update_speed():
    domTree = xml.etree.ElementTree.parse(route_file4)
    root = domTree.getroot()

    inputs = root.findall("vType")

    for item in inputs:
        item.set('maxSpeed', '20.00')
    domTree.write('rou789.xml')


"解析xml速度分布"
def speed_distr():

    x_spd = []
    y_lik = []

    path_file1 = 'tripinfo/tripinfo_agent6000.xml'
    path_file2 = 'tripinfo/tripinfo_no_agent6000.xml'

    domTree = xml.etree.ElementTree.parse(path_file1)
    root = domTree.getroot()
    inputs = root.findall("tripinfo")

    domTree2 = xml.etree.ElementTree.parse(path_file2)
    root2 = domTree2.getroot()
    inputs2 = root2.findall("tripinfo")

    _cnt, cnt = 0,0

    # 排除停车车辆
    data = pd.read_csv('veh_log_stop.csv')
    stop_veh = data['vehID']

    for item,item2 in zip(inputs, inputs2):
        if item.get('id') in stop_veh:
            continue
        dis = float(item.get('duration'))-float(item2.get('duration'))
        _cnt += dis / float(item2.get('duration')) if dis > 0 else 0
        cnt += 1 if dis > 0 else 0
        x_spd.append(dis)

    x_spd = np.array(x_spd)
    # print(x_spd.min())
    # print(x_spd.max())
    # print(x_spd.mean())
    # print(np.var(x_spd))

    # f = Fitter(x_spd, distributions=['uniform'])  # 创建Fitter类
    # f.fit()  # 调用fit函数拟合分布
    # # f.summary(Nbest=1, lw=2, plot=True, method='sumsquare_error')  # 输出拟合结果
    # print(f.fitted_param['uniform'])

    plt.hist(x_spd, bins=300, range=(10, 310))
    # plt.savefig('456.png')
    plt.show()


    print(_cnt/cnt)


def create_stop_vehicle():

    # 读取发送车辆
    domTree = xml.etree.ElementTree.parse(route_file3)
    root = domTree.getroot()

    inputs = root.findall("vehicle")
    data = pd.DataFrame(columns=['vehID', 'in_time', 'out_time'])
    for item in inputs:
        if random.random() < 0.1:
            item.set('color', '1,1,0')
            data.loc[len(data.index)] = [item.get('id'), float(item.get('depart')),
                                         float(item.get('depart')) + random.randint(300, 900)]

    data.to_csv('veh_log_stop.csv', header=1, index=False)
    domTree.write(route_file3)



# create_file()
# update_file()
# sort_vehicle()
create_stop_vehicle()

# update_speed()

# speed_distr()

# sampleno = 4303
# totalno = 0
# sampleno_epoch = sampleno
#
# while totalno < sampleno:
#     sample_result = np.random.normal(10.5546875, 0.4, sampleno_epoch)
#     result2 = np.array(sample_result)
#     result2 = result2[(result2 >= 6) & (result2 <= 17)]
#     totalno = len(result2)
#     sampleno_epoch = sampleno_epoch + 100
#
# plt.hist(result2, bins=900, range=(6, 17))
#
# plt.savefig('adapt.png')
# plt.show()
#
#
