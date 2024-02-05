"""
    静态停车
    目标车：vehID
    估计停车时间计算：（out_time - in_time）-trip_info.duration
    停车点 随机插入
"""
import pandas as pd
import traci
from traci import TraCIException
from random import randint

# 估计 target 目标车辆的停车时间

target = pd.read_csv('conf/veh_log_test.csv')
trip_info = pd.read_csv('record/no_stop_test_tripinfo.csv', sep=';')
target_stop = {}
stop_dict = dict(zip(target['vehID'].values, [False for i in range (len(target.values))]))
status_dict = dict(zip(target['vehID'].values, [False for i in range (len(target.values))]))
sim_dict = {} # 产生一个随机停车点
time_stop = {}

MAX_DIS_PARKING = 20

for idx, data in target.iterrows():
    a = trip_info.loc[trip_info['tripinfo_id'] == data['vehID'], 'tripinfo_duration']
    if not a.empty:
        target_stop[data['vehID']] = data['out_time'] - data['in_time'] - a.values[0] if data['out_time'] - data['in_time'] - a.values[0] > 0 else 0
        sim_dict[data['vehID']] = data['in_time'] + randint(0, a.values[0])
        print(data['vehID'], data['out_time'] - data['in_time'], target_stop[data['vehID']])
#
df = pd.DataFrame.from_dict(sim_dict, orient='index',columns=['depart'])
df = df.reset_index().rename(columns = {'index':'vehid'})
df.to_csv('sim_dict.csv', index=False)

# df = pd.read_csv('sim_dict.csv')
# sim_dict = dict(zip(df['vehid'],df['depart']))


traci.start(["sumo-gui", "-c", 'conf/aofeng.sumocfg'])


def tooCloseToPark(vehID):
    return traci.lane.getLength(traci.vehicle.getLaneID(vehID)) \
           - traci.vehicle.getLanePosition(vehID) > MAX_DIS_PARKING and \
           traci.vehicle.getLanePosition(vehID) > MAX_DIS_PARKING

def tryStopVeh(vehID, spd, currentRd, lanePos, routeID, routeIdx, dration, laneId):
    # 计划停车，进入换道

    if spd <= 0:
        return True
    if traci.vehicle.getLaneID(vehID).split('_')[1] != '0':
        traci.vehicle.changeLane(vehID, 0, duration=2)
        return False
    elif spd > 0.1:
        traci.vehicle.slowDown(vehID, 0, 1)
        return False
    else:
        edges = traci.route.getEdges(routeID)
        newRoute = edges[routeIdx:]
        try:
            if currentRd in edges and tooCloseToPark(vehID) and \
                    traci.vehicle.isStopped(vehID) is False:
                traci.vehicle.insertStop(vehID, 0, currentRd, pos=lanePos, laneIndex=0)
                traci.vehicle.setRoute(vehID, newRoute)
                return True
            else:
                return False
        except TraCIException as e:
            print(e)
            return False


def resumeVeh(vehID):
    traci.vehicle.resume(vehID)
    return True


if __name__ == '__main__':
    step = 0
    status_stop = False
    been_stop = False
    time_stop = {}
    tmp_count = 0

    target_onlint = set()
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            now = traci.simulation.getTime()
            veh_online = set(traci.vehicle.getIDList())
            target = set(target_stop.keys())
            target_online = target.intersection(veh_online)

            for tar in target_online:
                newRd = traci.vehicle.getRoadID(tar)
                spd = traci.vehicle.getSpeed(tar)
                laneLen = traci.lane.getLength(traci.vehicle.getLaneID(tar))
                timeLoss = traci.vehicle.getTimeLoss(tar)
                lanePos = traci.vehicle.getLanePosition(tar)
                laneId = traci.vehicle.getLaneID(tar)
                routeID = traci.vehicle.getRouteID(tar)
                routeIdx = traci.vehicle.getRouteIndex(tar)
                currentRd = traci.vehicle.getRoadID(tar)

                if (not stop_dict[tar]) and (not status_dict[tar]) and now >= sim_dict[tar]:
                    if tryStopVeh(tar, spd, currentRd, lanePos, routeID, routeIdx, target_stop[tar], laneId):
                        stop_dict[tar] = True
                        status_dict[tar] = True
                        time_stop[tar] = now
                if traci.vehicle.isStopped(tar) and status_dict[tar] and now >= time_stop[tar] + target_stop[tar]:
                    status_dict[tar] = False
                    resumeVeh(tar)
            step += 1

    except Exception as e:
        traci.close()
        print(e)
    else:
        traci.close()
        print("safely ended.")









