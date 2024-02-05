from itertools import count

import datetime as dt
import pandas as pd
import torch
import Simulation as Simulation

import strategies
from VehAgent import VehAgent
from strategies import Optimizer, ActionSelector
from tqdm import tqdm
import time


if __name__ == '__main__':
    # inputs
    sumo_config = './conf/aofeng.sumocfg'
    sumo_rd_net_url = './conf/xuancheng1116_6.net.xml'
    veh_logs_train = './conf/veh_log_bk.csv'
    veh_logs_test = './conf/veh_log_test'
    batch_size = 5

    name = dt.datetime.now().strftime("%Y%m%d%H%M")

    # outputs
    best_model_path = 'model/aofeng/best_model.pt'
    saved_network_format = 'model/aofeng/%s_epoch_model.pt'
    performance_csv = 'result/aofeng/performance.csv'
    rand_trace_csv = 'result/aofeng/rand_trace_%s.csv' % name
    eval_trace_csv = 'result/aofeng/eval_trace_%s.csv' % name

    if torch.cuda.is_available():
        num_episodes = 1000
    else:
        num_episodes = 100

    # Get number of actions from gym action space
    # n_actions = env.action_space.n
    # 0 drive; 1 stop 0.5 min; 2 stop 1 min; 3 stop 5 min
    n_actions = 4

    # Get the number of state observations
    # dis_to_end, timeRemain, gap, leading_spd, spd, avg_spd
    n_observations = len(Simulation.Observation._fields) - 2

    # initial functional parts:
    optimizer = Optimizer(n_observations, n_actions, best_model_path)
    act_selector = ActionSelector(n_actions, optimizer)
    sim = Simulation.Simulation(sumo_config, sumo_rd_net_url, veh_logs_train, need_gui=False)
    sim.connectSumo()

    with open(rand_trace_csv, 'w+') as w:
        w.write(VehAgent.printHeaders())
    w.close()
    with open(eval_trace_csv, 'w+') as w:
        w.write(VehAgent.printHeaders())
    w.close()

    steps_done = 0
    g_eps_threshold = 0

    r_rate = act_selector.eps_threshold()
    i_episode = 0

    while i_episode < num_episodes:
        start_time = time.time()
        i_episode += 1
        with tqdm(total=batch_size)as pbar:
            for batch in range(batch_size):
                act_selector.steps_done += 1
                sim.resetSim()
                vehSample_tup, _, _ = sim.runUntilTargetShow()
                veh_agents_dict = VehAgent.generateFromVSamples(vehSample_tup, sim)
                veh_agents_arrived = []
                for t in count():
                    import time
                    start = time.time()
                    # 根据初始状态产生新动作并执行新动作
                    for agent in veh_agents_dict.values():
                        if agent.canAct():
                            act = act_selector.select_action(agent, update_experience=False)
                            agent.applyAction(act, strategies.PERIOD)
                        else:
                            agent.keep()
                    # 环境推进

                    vehSamples_tdn, terminated, truncated = sim.runUntilTargetShow()

                    new_veh_agents = []
                    for new_sample in vehSamples_tdn:
                        vid = new_sample.vehID
                        if vid in veh_agents_dict:
                            agent: VehAgent = veh_agents_dict[vid]
                            agent.tryUpdateStates(new_sample)
                        elif new_sample.state is not None:
                            # manage new_veh
                            new_veh_agents.append(VehAgent(new_sample, sim))
                    # 收集RL样本
                    for potential in veh_agents_dict.values():
                        rl_res = potential.tryGetRLSample()
                        if rl_res is not None:
                            optimizer.push_to_memory(rl_res)

                    # remove arrived vehicles
                    to_rm = [item[0] for item in veh_agents_dict.items() if item[1].is_finished()]
                    for vid in to_rm:
                        va = veh_agents_dict.pop(vid)
                        veh_agents_arrived.append(va)
                    # add new loaded vehicles
                    for va in new_veh_agents:
                        veh_agents_dict[va.veh_id] = va

                    # 处理仿真完成事件
                    done = terminated or truncated
                    end = time.time()
                    if end - start > 1:
                        print()
                    if done:
                        sim.close()
                        if terminated:
                            # Perform one step of the optimization (on the policy network)
                            loss = optimizer.optimize_model()
                            optimizer.update_target_net()
                            randErr = sim.calPerformance()
                            sid = "random_%s_%s" % (i_episode, batch)
                            with open(rand_trace_csv, 'a+') as w:
                                for va in veh_agents_arrived:
                                    err = sim.individualErr(va.veh_id)
                                    w.write(va.printRows(sim_id=sid, veh_id=str(va.veh_id), err=err, total_err=randErr))
                            w.close()
                            pbar.set_postfix({
                                'loss': '%.4f' % (loss if loss is not None else 0),
                                'random sel rate': '%.3f' % (r_rate),
                                'rand err': '%.3f' % randErr
                            })
                        pbar.update(1)
                        break
            pbar.close()
        optimizer.save_network_to(saved_network_format % i_episode)
        sid_ = "eval_%s" % i_episode
        evalErr = optimizer.eval_sim(optimizer, sim, act_selector, trace_csv=eval_trace_csv, sim_id=sid_)
        best_err = pd.read_csv(performance_csv, header=0)
        if best_err['err'][0] > evalErr:
            best_err['err'][0] = evalErr
            best_err.to_csv(performance_csv, index=False)
            optimizer.save_network_to(best_model_path)

        r_rate = act_selector.eps_threshold()
        print("i_episode %s : valid err %s" % (i_episode, evalErr))




