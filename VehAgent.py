import numpy as np
from typing import List
from collections import namedtuple

import torch

import strategies
from Simulation import Simulation
import Simulation as sim
import traci as traci
# import traci
WatchVar = namedtuple('WatchVar', ('time', 'vs_before', 'vs_after', 'action', 'reward'))


class VehAgent(object):

    def __init__(self, sample: sim.VehSample, sim: Simulation):
        self.next_action_time = 0.0
        self.before = None
        self.current = sample
        self.sim = sim
        self.veh_id = sample.vehID
        self.applied_action = np.NaN
        self.is_halted = False
        self.been_halted = False
        # watch params
        self.watch_var = []

    def advance(self, new_sample: sim.VehSample):
        self.before = self.current
        self.current = new_sample

    @staticmethod
    def currentTime():
        return traci.simulation.getTime()

    def advanceActionTime(self, duration):
        self.next_action_time = self.currentTime() + duration

    def is_finished(self):
        return self.current.state is None

    def canAct(self):
        # 时间到达下一个决断时刻，或已经完成仿真
        return self.next_action_time <= self.currentTime()

    def keep(self):
        currentSpd = traci.vehicle.getSpeed(self.veh_id)
        if self.is_halted:
            if currentSpd > 0:
                currentRd = traci.vehicle.getRoadID(self.veh_id)
                lanePos = traci.vehicle.getLanePosition(self.veh_id)
                routeID = traci.vehicle.getRouteID(self.veh_id)
                routeIdx = traci.vehicle.getRouteIndex(self.veh_id)
                self.sim.tryStopVeh(self.veh_id, currentSpd, currentRd, lanePos, routeID, routeIdx)
        else:
            if traci.vehicle.isStopped(self.veh_id):
                self.sim.resumeVeh(self.veh_id)

    def applyAction(self, action, period):
        self.applied_action = action

        currentRd = traci.vehicle.getRoadID(self.veh_id)
        lanePos = traci.vehicle.getLanePosition(self.veh_id)
        routeID = traci.vehicle.getRouteID(self.veh_id)
        routeIdx = traci.vehicle.getRouteIndex(self.veh_id)
        currentSpd = traci.vehicle.getSpeed(self.veh_id)
        # act action
        if action != -1:
            need_to_park = True
            if action == 1:
                duration = 1 * period
            elif action == 2:
                duration = 2 * period
            elif action == 3:
                duration = 5 * 2 * period
            else:
                duration = 1 * period
                need_to_park = False

            if self.is_halted:
                # 状态：停车中
                if need_to_park:
                    # 继续停车
                    # 考虑车速可能不为0，所以继续执行tryStopVeh。
                    self.is_halted = True
                    self.sim.tryStopVeh(self.veh_id, currentSpd, currentRd, lanePos, routeID, routeIdx)
                else:
                    # 新动作为行驶，且当前已停车，按要求启动
                    self.is_halted = False
                    self.been_halted = True
                    if traci.vehicle.isStopped(self.veh_id):
                        self.sim.resumeVeh(self.veh_id)
            else:
                # 状态：行驶中
                if need_to_park:
                    self.is_halted = True
                    self.sim.tryStopVeh(self.veh_id, currentSpd, currentRd, lanePos, routeID, routeIdx)
                else:
                    self.is_halted = False

            if self.been_halted:
                # 已完成停车的，下个决断时刻为完成出行后
                # 因为“从停车状态重新启动”这个动作的后果就是保持运行直至完成
                self.advanceActionTime(np.inf)
            else:
                self.advanceActionTime(duration)

    def tryUpdateStates(self, new_sample: sim.VehSample):
        if self.current.state is None and new_sample.state is None:
            print("DEBUG: duplicated finished state.")
        if self.canAct():
            self.advance(new_sample)
        if new_sample.state is None:
            self.advance(new_sample)

    def tooCloseToPark(self):
        return self.sim.tooCloseToPark(self.veh_id)

    @staticmethod
    def generateFromVSamples(vsamples: List[sim.VehSample], simulation: Simulation):
        ans = [VehAgent(v, simulation) for v in vsamples]
        ans = dict(zip([a.veh_id for a in ans], ans))
        return ans

    def canEval(self):
        return self.canAct() or self.is_finished()

    def printHist(self):
        acts = [var.action for var in self.watch_var]
        print("acts: %s " % acts)
        time = [var.time for var in self.watch_var]
        print("time: %s " % time)
        pos = [var.vs_before.state.dis_to_end for var in self.watch_var]
        print("positions: %s " % pos)
        estimation = [var.vs_before.eval.time_err for var in self.watch_var]
        print("err_eval: %s " % estimation)
        reward = [var.reward for var in self.watch_var]
        print("reward: %s " % reward)

    def printRows(self, sim_id='sim_unknown', veh_id='id_unknown', err=0, total_err=0):
        rows = ''
        for var in self.watch_var:
            if var.vs_after.state is None:
                dis_after = 0
            else:
                dis_after = var.vs_after.state.dis_to_end
            try:
                row = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
                      % (sim_id,
                         veh_id,
                         var.time,
                         var.action,
                         var.vs_before.state.dis_to_end,
                         var.vs_before.eval.time_err,
                         var.vs_before.eval.portion,
                         dis_after,
                         var.vs_after.eval.time_err,
                         var.vs_before.eval.portion,
                         var.reward,
                         err,
                         total_err)
                rows += row + '\r'
            except AttributeError as e:
                print(e)
        return rows

    @staticmethod
    def printHeaders():
        return "sim,veh,time,action,dis_before,err_before,port_before,dis_after,err_after,port_after,reward,err," \
               "total_err\r"

    def tryLog(self):
        if self.canEval() and self.canCalReward():
            reward = self.sim.calReward(self.before, self.current)
            watch_var = WatchVar(self.currentTime(), self.before, self.current, self.applied_action, reward)
            self.watch_var.append(watch_var)

    def tryGetRLSample(self):
        if self.canEval() and self.canCalReward():
            w = self.current.state.p_time if self.current.state is not None else 1.0
            reward = self.sim.calReward(self.before, self.current) * np.max((w, 1.0))
            watch_var = WatchVar(self.currentTime(), self.before, self.current, self.applied_action, reward)
            self.watch_var.append(watch_var)
            state = torch.tensor(self.before.state[2:], dtype=torch.float32, device=strategies.device).unsqueeze(0)
            action = torch.tensor([self.applied_action], device=strategies.device).unsqueeze(0)
            if self.current.state is None:
                next_state = None
            else:
                next_state = torch.tensor(self.current.state[2:],
                                          dtype=torch.float32,
                                          device=strategies.device) \
                    .unsqueeze(0)
            reward = torch.tensor(reward, device=strategies.device).unsqueeze(0)
            return strategies.Transition(state, action, next_state, reward)
        else:
            return None

    def canCalReward(self):
        return (self.before.eval is not None) \
            and (self.current.eval is not None) \
            and (self.before.state is not None)
