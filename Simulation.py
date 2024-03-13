import traci as traci

from collections import namedtuple
import pandas as pd
import numpy as np

from lxml import etree as ET
from traci.exceptions import TraCIException

VehSample = namedtuple('VehSample', ('vehID', 'state', 'eval'))
Evaluation = namedtuple('Evaluation', ('time_err', 'portion'))
Observation = namedtuple('Observation', ('parking', 'been_parked', 'dis_to_end', 'time_to_end', 'spd', 'trt', 'p_time'))
TargetLog = namedtuple('TargetLog', ('vehID', 'in_time', 'out_time'))
VehStatus = namedtuple('VehStatus', ('vehID', 'speed', 'time', 'pos', 'leader_veh'))


class Simulation:
    __need_gui: bool
    __config: str
    __target_logs: dict
    __target_veh_ids: set
    # __veh_entrance: dict
    # __veh_exit: dict
    __connections: dict
    __lanes: dict
    # __route_len: dict
    # __now: float
    # __stopped_from: dict
    # __restart_from: dict
    max_spd = 20
    MAX_PARKING_DURATION = 900
    MAX_DIS_PARKING = 20

    def __init__(self, config, rd_net_url, target_logs_csv, need_gui=False, label='default'):
        self.need_gui = need_gui
        self.config = config
        self.__target_logs = dict()
        self.stop_env = True if 'stop' in target_logs_csv else False
        if target_logs_csv is not None:
            df_logs = pd.read_csv(target_logs_csv)
            self.__target_veh_ids = set(df_logs.vehID.values.tolist())
            for index, row in df_logs.iterrows():
                self.__target_logs[row['vehID']] = TargetLog(row['vehID'], row['in_time'], row['out_time'])
        self.__connections = dict()
        self.__lanes = dict()
        tree = ET.parse(rd_net_url)
        for c in tree.findall('connection'):
            if 'via' in c.attrib:
                key = (c.attrib['from'], c.attrib['to'])
                if key not in self.__connections:
                    self.__connections[key] = list()
                self.__connections[key].append(c.attrib['via'])
        for e in tree.findall('edge'):
            if 'function' not in e.attrib or e.attrib['function'] != 'internal':
                self.__lanes[e.attrib['id']] = list()
                for l in e.findall('lane'):
                    self.__lanes[e.attrib['id']].append(l.attrib['id'])
        self.__route_len = dict()
        self.__veh_entrance = dict()
        self.__veh_exit = dict()
        self.__stopped_from = dict()
        self.__restart_from = dict()
        self.__now = 0
        self.label = label
        self.connection = None

    def connectSumo(self):
        if self.connection is None:
            if self.need_gui:
                traci.start(["sumo-gui", "-c", self.config, '--delay', '1', '--time-to-teleport', '-1', '-W', '-S', '-Q'],
                            label=self.label)
            else:
                traci.start(["sumo", "-c", self.config, '--time-to-teleport', '-1', '-W'], label=self.label)
        self.connection = traci.getConnection(self.label)
        now = self.connection.simulation.getTime()
        self.__now = now

    def reset_sim_mem(self):
        self.__route_len = dict()
        self.__veh_entrance = dict()
        self.__veh_exit = dict()
        self.__stopped_from = dict()
        self.__restart_from = dict()

    def resetSim(self, valid=False):
        self.connection.close()
        self.reset_sim_mem()
        self.connectSumo()
        return self.runUntilTargetShow()

    def runUntilTargetShow(self):

        no_target = True
        online_target = set()
        offline_target = set()

        # while self.connection.simulation.getMinExpectedNumber() > 0 and no_target:
        while self.connection.simulation.getTime() < self.connection.simulation.getEndTime() and no_target:
            veh_online, veh_offline = self.step()
            online_target = self.__target_veh_ids.intersection(veh_online)
            offline_target = self.__target_veh_ids.intersection(veh_offline)
            no_target = ((online_target == set()) and (offline_target == set()))

        # 断电测试
        # if '221888670#10__15.46' in veh_online:
        #     a = 1

        terminated = self.connection.simulation.getTime() < self.connection.simulation.getEndTime()
        truncated = (not traci.isLoaded())
        veh_output = []

        for v in online_target:
            veh_output.append(VehSample(v, self.observe(v), self.evalOnline(v)))
        for v in offline_target:
            veh_output.append(VehSample(v, None, self.evalOffline(v)))

        return veh_output, terminated, truncated

    def run_until_record_in_out(self):
        veh_output, terminated, truncated = self.runUntilTargetShow()
        veh_online = self.connection.vehicle.getIDList()
        veh_offline = self.connection.simulation.getArrivedIDList()
        return veh_output, terminated, truncated, veh_online, veh_offline

    def calReward(self, sample_0: VehSample, sample_1: VehSample):
        return self.calRewardFromEval(sample_0.eval, sample_1.eval)

    @staticmethod
    def calRewardFromEval(eval_0: Evaluation, eval_1: Evaluation):
        w = 1e-2
        alpha = 1.0
        beta = 2.0
        eta = eval_1.portion
        delta_err = eval_1.time_err - eval_0.time_err
        abs_err = np.max((1e-9, np.abs(eval_0.time_err)))
        reward = (-1 * delta_err)
        return reward

    def step(self):
        self.connection.simulationStep()
        now = self.connection.simulation.getTime()
        self.__now = now

        # Returns a list of all objects in the network.
        veh_online = self.connection.vehicle.getIDList()

        # Returns a list of ids of vehicles which arrived
        veh_offline = self.connection.simulation.getArrivedIDList()

        new_veh = set(veh_online).difference(set(self.__veh_entrance.keys()))
        pass_veh = set(veh_offline)
        for v in new_veh:
            self.__veh_entrance[v] = now
        for v in pass_veh:
            self.__veh_exit[v] = now

        return veh_online, veh_offline

    def observe(self, vehID):
        """
        vehID state
        :param vehID:
        :return: [dis_to_end, self.timeRemain(vehID), gap, leading_spd, spd, mileage/t_eclipse]
        """

        # Return the leading vehicle id together with the distance
        leader = self.connection.vehicle.getLeader(vehID)

        spd = self.connection.vehicle.getSpeed(vehID)
        routeID = self.connection.vehicle.getRouteID(vehID)
        routeIdx = self.connection.vehicle.getRouteIndex(vehID)
        laneID = self.connection.vehicle.getLaneID(vehID)
        laneLen = self.connection.lane.getLength(laneID)

        # # Returns the distance to the starting point like an odometer.
        mileage = self.connection.vehicle.getDistance(vehID)

        # The position of the vehicle along the lane measured in m.
        lanePos = self.connection.vehicle.getLanePosition(vehID)

        if leader is None:
            # max speeding
            gap = laneLen - lanePos
            leading_spd = Simulation.max_spd
        else:
            # leader speeding
            leaderID = leader[0]
            # min gap not included in the follows, gap >= 0
            gap = leader[1]
            leading_spd = self.connection.vehicle.getSpeed(leaderID)

        # get running time of line
        t_eclipse = self.timeEclipse(vehID)

        # Remaining distance (percentage)
        dis_to_end = self.disToEnd(vehID)
        routeID = self.connection.vehicle.getRouteID(vehID)
        whole_end = self.wholeRouteLen(routeID)
        dis_to_end = 0

        # Remaining time (percentage)
        # time_remain = 0
        # if self.__target_logs[vehID].out_time > self.__now:
        #     time_remain = (self.__now - self.__target_logs[vehID].in_time) \
        #         / (self.__target_logs[vehID].out_time - self.__target_logs[vehID].in_time)
        


        avg_spd = 0
        # if t_eclipse > 0:
        #     avg_spd = mileage / t_eclipse

        # 1 在停车状态 0 在行驶状态
        tmp1 = 1 if (vehID in self.__stopped_from) and (vehID not in self.__restart_from) else 0
        # 1 已完成停车控制 0 未完成停车控制
        tmp2 = 1 if vehID in self.__restart_from else 0

        # proportion of expect_time in expect_time + use_time
        trt = self.est_travel_time(routeID, routeIdx, laneID, lanePos)

        time_remain = (self.__target_logs[vehID].out_time - self.__target_logs[vehID].in_time) - self.getOnlineParkTime(vehID) - trt

        p_time = self.getOnlineParkTime(vehID)
        # return [tmp1, tmp2, dis_to_end, self.timeRemain(vehID), spd, trt, p_time]
        return Observation(tmp1, tmp2, dis_to_end, time_remain, spd, trt, p_time)

    def checkParkTimeOver(self, vehID):
        return self.getOnlineParkTime(vehID) > Simulation.MAX_PARKING_DURATION

    def tooCloseToPark(self, vehID):
        return self.connection.lane.getLength(self.connection.vehicle.getLaneID(vehID)) \
            - self.connection.vehicle.getLanePosition(vehID) < self.MAX_DIS_PARKING and \
               self.connection.vehicle.getLanePosition(vehID) < self.MAX_DIS_PARKING

    def getOnlineParkTime(self, vehID):
        if vehID in self.__stopped_from:
            t_end = self.__restart_from[vehID] if vehID in self.__restart_from else self.__now
            accumulated_park_time = t_end - self.__stopped_from[vehID]
        else:
            accumulated_park_time = 0
        return accumulated_park_time

    def stoppedFrom(self):
        return self.__stopped_from

    def evalOnline(self, vehID):
        # if self.timeEclipse(vehID) == 0:
        #     spd = self.connection.vehicle.getSpeed(vehID)
        # else:
        #     spd = self.connection.vehicle.getDistance(vehID) / self.timeEclipse(vehID)
        # return -1.*np.abs(self.disToEnd(vehID) / self.timeRemain(vehID) - spd)

        routeID = self.connection.vehicle.getRouteID(vehID)
        routeIdx = self.connection.vehicle.getRouteIndex(vehID)
        laneID = self.connection.vehicle.getLaneID(vehID)
        lanePos = self.connection.vehicle.getLanePosition(vehID)
        mileage_cap = self.est_mileage(routeID, routeIdx, lanePos)
        # mileage = self.connection.vehicle.getDistance(vehID)
        r_len = self.wholeRouteLen(routeID)
        accumulated_park_time = self.getOnlineParkTime(vehID)
        t_out_real = self.__target_logs[vehID].out_time
        t_exp = self.est_travel_time(routeID, routeIdx, laneID, lanePos)
        t_wait = t_out_real - self.__now - t_exp
        t_in_real = self.__target_logs[vehID].in_time
        time_err = np.abs(t_out_real - t_in_real - (self.__now - self.__veh_entrance[vehID] + t_exp))
        road_part = min(mileage_cap / r_len, 1)
        # exp = road_part + min(accumulated_park_time / t_wait, 1)
        # return
        return Evaluation(time_err, road_part)

    def evalOffline(self, vehID):
        # rid = self.__veh_route[vehID]
        # journey = self.wholeRouteLen(rid)
        # travelTime = self.__veh_exit[vehID] - self.__veh_entrance[vehID]
        # # todo: 将所有车辆的都添加到log.cvs里面，且补充routeID列，不需要__veh_route单独记录
        # log = self.__target_logs[vehID]
        # loggedTravelTime = log.out_time - log.in_time
        # return -1.*np.abs(journey/loggedTravelTime - journey/travelTime)

        if vehID in self.__stopped_from:
            t_end = self.__restart_from[vehID] if vehID in self.__restart_from else self.__veh_exit[vehID]
            accumulated_park_time = t_end - self.__stopped_from[vehID]
        else:
            accumulated_park_time = 0
        log = self.__target_logs[vehID]
        loggedTravelTime = log.out_time - log.in_time
        travelTime = self.__veh_exit[vehID] - self.__veh_entrance[vehID]
        # road_part = 1
        # exp = road_part + min(accumulated_park_time / (loggedTravelTime - (travelTime - accumulated_park_time)), 1)
        # return np.exp(exp)
        return Evaluation(np.abs(travelTime - loggedTravelTime), 1)

    def individualErr(self, vehID):
        travelTime = self.__veh_exit[vehID] - self.__veh_entrance[vehID]
        log = self.__target_logs[vehID]
        loggedTravelTime = log.out_time - log.in_time
        return abs(travelTime - loggedTravelTime)

    def calPerformance(self):
        # todo: 需要考虑未成功完成出行的目标车
        err = np.mean([self.individualErr(x) for x in self.__target_veh_ids
                       if x in self.__veh_exit and x in self.__veh_entrance])
        return err

    def timeEclipse(self, vehID):
        return self.__now - self.__veh_entrance[vehID]

    def timeRemain(self, vehID):
        return self.__target_logs[vehID].out_time - self.__now

    def getTraveltime(self, edge_id):
        n_veh = self.connection.edge.getLastStepVehicleNumber(edge_id)
        n_halt = self.connection.edge.getLastStepHaltingNumber(edge_id)
        if (n_veh - n_halt) > 0 :
            ans = self.connection.edge.getTraveltime(edge_id)
        else:
            ans = self.edgeLen(edge_id) / self.max_spd
        return ans

    def est_travel_time(self, routeID, routeIdx, laneID, lanePos):
        edges = self.connection.route.getEdges(routeID)
        _sum = 0
        idx = edges.index(edges[routeIdx])
        for i in range(idx, len(edges)):
            _sum += self.getTraveltime(edges[i])
        lane_len = self.connection.lane.getLength(laneID)
        offset = self.getTraveltime(edges[idx]) * lanePos / lane_len
        return _sum - offset

    def est_mileage(self, routeID, routeIdx, lanePos):
        return self.routeLen(routeID, routeIdx) + lanePos

    def edgeLen(self, edge_id):
        lanes = self.__lanes[edge_id]
        len = np.average([self.connection.lane.getLength(lane) for lane in lanes])
        return len

    def routeLen(self, routeID, routeIdx):
        edges = self.connection.route.getEdges(routeID)
        _len = 0
        idx = edges.index(edges[routeIdx])
        if idx > 0:
            for i in range(idx - 1):
                ckey = (edges[i], edges[i + 1])
                # todo: 某些edge之间不存在connection 待查明
                if ckey in self.__connections:
                    conns = self.__connections[ckey]
                    _len += np.average([self.connection.lane.getLength(conn) for conn in conns])
        # add avg_len_lanes[normal]
        for i in range(idx):
            _len += self.edgeLen(edges[i])
        return _len

    def wholeRouteLen(self, routeID):
        if routeID not in self.__route_len:
            edges = self.connection.route.getEdges(routeID)
            tmp_len = self.edgeLen(edges[-1])
            self.__route_len[routeID] = \
                self.routeLen(routeID, -1) + tmp_len
        return self.__route_len[routeID]

    def disToEnd(self, vehID):
        lanePos = self.connection.vehicle.getLanePosition(vehID)
        routeID = self.connection.vehicle.getRouteID(vehID)
        routeIdx = self.connection.vehicle.getRouteIndex(vehID)
        mileage_cap = self.est_mileage(routeID, routeIdx, lanePos)
        journey = self.wholeRouteLen(routeID)
        return journey - mileage_cap

    # def timeRemain(self, vehID):
    #     t_expected = self.__veh_exit[vehID]
    #     return t_expected - self.__now

    def tryStopVeh(self, vehID, spd, currentRd, lanePos, routeID, routeIdx):
        # if '221888670#10__15.46' == vehID:
        #     a = 1
        if vehID not in self.__stopped_from and self.connection.vehicle.getStopState(vehID) == 1:
            self.__stopped_from[vehID] = self.__now
        if spd <= 0:
            return True
        if int(self.connection.vehicle.getLaneID(vehID).split('_')[1]) > 1:
            self.connection.vehicle.changeLane(vehID, 1, duration=2)
            return False
        if self.connection.vehicle.getLaneID(vehID).split('_')[1] != '0':
            if spd > 3:
                self.connection.vehicle.slowDown(vehID, 1, 1)
            if len(self.connection.vehicle.getRightLeaders(vehID)) == 0 or self.connection.vehicle.getRightLeaders(vehID)[0][1] > 10:
                #  0 车道上具有足够的空间提供减速或者变道
                self.connection.vehicle.changeLane(vehID, 0, duration=2)
            return False
        elif spd > 0.1:
            self.connection.vehicle.slowDown(vehID, 0, 1)
            return False
        else:
            edges = self.connection.route.getEdges(routeID)
            newRoute = edges[routeIdx:]
            try:
                if self.connection.vehicle.isStopped(vehID) is False and currentRd in edges:
                    self.connection.vehicle.insertStop(vehID, 0, currentRd, pos=lanePos, laneIndex=0)
                    self.connection.vehicle.setRoute(vehID, newRoute)
                    return True
                else:
                    return False
            except TraCIException as e:
                print(e)
                return False

    def resumeVeh(self, vehID):
        if vehID not in self.__restart_from:
            self.__restart_from[vehID] = self.__now
        try:
            self.connection.vehicle.resume(vehID)
        except Exception:
            print("DEBUG resumeVeh exception")

    def except_travel_time(self, vehID):
        return self.__target_logs[vehID].out_time - self.__target_logs[vehID].in_time

    def set_veh_sped(self, vehID, sped):
        try:
            self.connection.vehicle.setSpeed(vehID, sped)
        except Exception:
            print('set speed error')

    def get_time(self):
        return self.connection.simulation.getTime()

    def get_neighbor_veh(self, veh_id, dis=50.0):
        if veh_id in self.connection.vehicle.getIDList():
            follower_veh = self.connection.vehicle.getFollower(veh_id, dis)
            leader_veh = self.connection.vehicle.getLeader(veh_id, dis)
            return leader_veh, follower_veh
        return None, None

    def get_veh_status(self, veh_id):
        if veh_id in self.connection.vehicle.getIDList():
            speed = self.connection.vehicle.getSpeed(veh_id)
            current = self.connection.simulation.getTime()
            pos = self.connection.vehicle.getLaneID(veh_id)
            leader_veh = self.connection.vehicle.getLeader(veh_id)
            return VehStatus(veh_id, speed, current, pos, leader_veh[0] if leader_veh is not None else '')
        else:
            return None

    def target_veh(self):
        return self.__target_logs

    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
                return True
        except TraCIException as e:
            return False



if __name__ == '__main__':
    sim = Simulation('./conf/aofeng.sumocfg', './conf/xuancheng1116_6.net.xml', './conf/veh_log_train.csv')
    sim.connectSumo()
    sim.runUntilTargetShow()
    # for v_id in sim._Simulation__target_veh_ids:
    #     print(sim.observe(v_id))
    # self.connectionclose(False)
