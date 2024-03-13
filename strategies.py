from itertools import count
from typing import List

import math
import os
from collections import namedtuple, deque
import random

import matplotlib
import numpy as np
import torch
import traci as traci
# import traci
from matplotlib import pyplot as plt
from torch import optim, nn

from Simulation import Simulation
from VehAgent import VehAgent, WatchVar
from dqn_network import DQN

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 0.1  # 0.1
EPS_END = 0.01
EPS_DECAY = 1e3
TAU = 0.005
LR = 1e-2
MEM_CAPACITY = 1000
PERIOD = 20
PARKING_MAX = 1000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if gpu is to be used
device = torch.device("cpu")


class Optimizer(object):
    def __init__(self, n_observations, n_actions, best_model_path: str):
        # try to load best model
        if os.path.exists(best_model_path):
            # load best model
            self.policy_net = torch.load(best_model_path, map_location='cpu').to(device)
            self.target_net = torch.load(best_model_path, map_location='cpu').to(device)
        else:
            self.policy_net = DQN(n_observations, n_actions).to(device)
            self.target_net = DQN(n_observations, n_actions).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            print(len(self.memory))
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + torch.squeeze(reward_batch)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # print("loss : %s" % loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                    1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_from_policy(self, input_vector):
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return self.policy_net(input_vector).max(1)[1].view(1, 1).item()

    def push_to_memory(self, mem):
        self.memory.push(mem)

    def save_network_to(self, saved_file: str):
        torch.save(self.policy_net, saved_file)

    @staticmethod
    def plot_performance(watch_vars: List[WatchVar], show_result=False, y_label='Duration', x_label='Episode'):
        plt.figure(1)
        y = [var.reward for var in watch_vars]
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(y)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    @staticmethod
    def eval_sim(self, sim: Simulation, act_selector, trace_csv=None, sim_id='Unknown'):
        sim.resetSim()
        vehSample_tup, _, _ = sim.runUntilTargetShow()
        veh_agents_dict = VehAgent.generateFromVSamples(vehSample_tup, sim)
        veh_agents_arrived = []
        for t in count():
            # 根据初始状态产生新动作并执行新动作
            for agent in veh_agents_dict.values():
                if agent.canAct():
                    act = act_selector.select_action(agent, randomly=False, update_experience=False)
                    agent.applyAction(act, PERIOD)
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
                else:
                    # manage new_veh
                    new_veh_agents.append(VehAgent(new_sample, sim))

            # logging
            for potential in veh_agents_dict.values():
                potential.tryLog()

            # manage new_veh
            # remove arrived vehicles
            to_rm = [item[0] for item in veh_agents_dict.items() if item[1].is_finished()]
            for vid in to_rm:
                va = veh_agents_dict.pop(vid)
                veh_agents_arrived.append(va)
            # add new loaded vehicles
            for va in new_veh_agents:
                veh_agents_dict[va.veh_id] = va

            if terminated or truncated:
                if terminated:
                    err = sim.calPerformance()
                    if trace_csv is not None:
                        with open(trace_csv, 'a+') as w:
                            for va in veh_agents_arrived:
                                single_err = sim.individualErr(va.veh_id)
                                w.write(va.printRows(sim_id=sim_id, veh_id=str(va.veh_id), err=single_err, total_err=err))
                        w.close()
                    return err
                else:
                    return np.NaN


class ActionSelector(object):
    def __init__(self, n_actions, optimizer: Optimizer, sim:Simulation):
        self.steps_done = 0
        self.n_actions = n_actions
        self.optimizer = optimizer
        self.sim = sim

    def eps_threshold(self):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
                        # (1.0 - np.min([self.steps_done / EPS_DECAY, 1.0]))
        return eps_threshold

    def select_action(self, veh: VehAgent, randomly=True, update_experience=True):
        withdraw = random.random()
        eps_threshold = self.eps_threshold()

        currentLane = self.sim.connection.vehicle.getLaneID(veh.veh_id)
        park_areas = ['64422', '63366']
        if veh.current.state is None:
            return -1
        elif currentLane.split('_')[0] not in park_areas:
            # 车辆所在道路没有停靠区域
            return 0
        elif currentLane == '' or (self.sim.connection.lane.getLength(currentLane) < 100) or veh.tooCloseToPark():
            return 0
        elif veh.been_halted:
            return 0
        elif veh.current.state.p_time > PARKING_MAX:
            return 0
        else:
            if randomly:
                if withdraw > eps_threshold:
                    if update_experience:
                        self.steps_done += 1
                    with torch.no_grad():
                        state = torch.tensor(veh.current.state[2:], dtype=torch.float32, device=device).unsqueeze(0)
                        return self.optimizer.generate_from_policy(state)
                else:
                    return random.randint(0, self.n_actions - 1)
            else:
                if update_experience:
                    self.steps_done += 1
                with torch.no_grad():
                    state = torch.tensor(veh.current.state[2:], dtype=torch.float32, device=device).unsqueeze(0)
                    if self.optimizer.generate_from_policy(state) == 0:
                        a = 1
                    return self.optimizer.generate_from_policy(state)


class ReplayMemory(object):

    def __init__(self, capacity=MEM_CAPACITY):
        self.memory = deque([], maxlen=capacity)
        self.best_memory = deque([], maxlen=capacity)
        self.thresold = 0
        self.rito = 0.5

    def push(self, mem):
        """Save a transition"""
        self.memory.append(mem)
        if mem.reward > self.thresold:
            self.best_memory.append(mem)
            if random.random() < 0.1:
                self.thresold = mem.reward

    def sample(self, batch_size):
        ran = random.sample(self.memory, int(batch_size * self.rito))
        ran.extend(random.sample(self.best_memory, int(batch_size * (1 - self.rito))))
        return ran

    def __len__(self):
        return len(self.best_memory)
