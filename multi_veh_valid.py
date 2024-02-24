import random
from itertools import count

import Simulation as Simulation

import strategies
from VehAgent import VehAgent
from strategies import Optimizer, ActionSelector

class Validation:

    __n_actions = 4

    def __init__(self, sumo_config, sumo_rd_net_url, sumo_target_logs_csv,
                 model_path, n_observations, n_actions):

        # initial functional parts:
        self.optimizer = Optimizer(n_observations, n_actions, model_path)
        self.act_selector = ActionSelector(n_actions, self.optimizer)
        self.sim = Simulation.Simulation(sumo_config, sumo_rd_net_url, sumo_target_logs_csv, need_gui=True)

    def run_sim(self):
        self.sim.connectSumo()
        vehSample_tup, _, _ = self.sim.runUntilTargetShow()
        veh_agents_dict = VehAgent.generateFromVSamples(vehSample_tup, self.sim)
        veh_agents_arrived = []
        for t in count():
            # 根据初始状态产生新动作并执行新动作
            for agent in veh_agents_dict.values():
                if agent.canAct():
                    act = self.act_selector.select_action(agent, randomly=False, update_experience=False)
                    agent.applyAction(act, strategies.PERIOD)
                else:
                    agent.keep()
            # 环境推进
            vehSamples_tdn, terminated, truncated = self.sim.runUntilTargetShow()
            new_veh_agents = []
            for new_sample in vehSamples_tdn:
                vid = new_sample.vehID
                if vid in veh_agents_dict:
                    agent: VehAgent = veh_agents_dict[vid]
                    agent.tryUpdateStates(new_sample)
                elif new_sample.state is not None:
                    # manage new_veh
                    new_veh_agents.append(VehAgent(new_sample, self.sim))

            # manage new_veh
            # remove arrived vehicles
            to_rm = [item[0] for item in veh_agents_dict.items() if item[1].is_finished()]
            for vid in to_rm:
                va = veh_agents_dict.pop(vid)
                veh_agents_arrived.append(va)
            # add new loaded vehicles
            for va in new_veh_agents:


                # compute average speed
                # sped = self.sim.wholeRouteLen('!' + va.veh_id) / (self.sim.except_travel_time(va.veh_id))
                # self.sim.set_veh_sped(va.veh_id, sped)

                veh_agents_dict[va.veh_id] = va

            # 处理仿真完成事件
            done = terminated or truncated
            if done:
                if terminated:
                    self.sim.close()
                    return True
                break
        return False


if __name__ == '__main__':

    stop_env = False

    # inputs
    sumo_config = './conf/aofeng.sumocfg'
    sumo_rd_net_url = './conf/xuancheng1116_6.net.xml'
    sumo_target_logs_csv = './conf/veh_log_stop.csv'
    # outputs
    best_model_path = 'model/aofeng/best_model.pt'

    n_actions = 4
    n_observations = len(Simulation.Observation._fields)

    valid = Validation(sumo_config, sumo_rd_net_url, sumo_target_logs_csv, best_model_path, n_observations, n_actions)
    print(valid.run_sim())

