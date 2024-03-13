import pickle
import queue
import threading
import time

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.formatters import Mesher

from calibration.Fullsim import Fullsim
from multi_veh_valid import Validation


sumo_config = 'conf/aofeng.sumocfg'
sumo_rd_net_url = 'conf/xuancheng1116_6.net.xml'
veh_logs_train = 'conf/veh_stop_truth.csv'


class Simthread(threading.Thread):
    exitFlag = 0
    queueLock = threading.Lock()
    workQueue = queue.Queue(0)
    threads = []
    thread_current_id = 1
    result = np.array([np.nan])

    def __init__(self):
        threading.Thread.__init__(self)
        self.threadID = Simthread.thread_current_id
        self.name = 'thread_' + str(self.threadID)
        # self.engine = Fullsim(sumo_config, sumo_rd_net_url, veh_logs_train, need_gui=False,
        #                       label='eng'+str(self.threadID))
        sumo_config = './conf/aofeng.sumocfg'
        sumo_rd_net_url = './conf/xuancheng1116_6.net.xml'
        sumo_target_logs_csv = './conf/veh_stop_truth.csv'
        # outputs
        best_model_path = 'model/aofeng/18_epoch_model.pt'

        n_actions = 4
        n_observations = 7
        self.engine = Validation(sumo_config, sumo_rd_net_url,
                                 sumo_target_logs_csv, best_model_path, n_observations, n_actions,
                                 label='eng'+str(self.threadID))
        Simthread.thread_current_id += 1

    def run(self):
        print("开启线程：" + self.name)
        while not Simthread.exitFlag:
            Simthread.queueLock.acquire()
            if not Simthread.workQueue.empty():
                head = Simthread.workQueue.get()
                Simthread.queueLock.release()
                self.process_sim(head)
            else:
                Simthread.queueLock.release()
            time.sleep(1)
        print("退出线程：" + self.name)

    def process_sim(self, head: dict):
        thread_ans = self.engine.run_with(head['para'])
        print(self.name + ' finished node ' + str(head))
        Simthread.result[head['loc']] = thread_ans

    @staticmethod
    def reset_results():
        Simthread.result = np.array([np.nan])


def create_threads():
    thread = Simthread()
    thread.start()
    Simthread.threads.append(thread)


def push_works(heads: list):
    Simthread.queueLock.acquire()
    for head in heads:
        Simthread.workQueue.put(head)
        # print("pushed " + str(head))
    Simthread.queueLock.release()


def end_threads():
    print('执行end')
    while not Simthread.workQueue.empty():
        pass
    Simthread.exitFlag = 1
    for t in Simthread.threads:
        t.join()
    return


def batch(para_array: np.array):
    (n, m) = np.shape(para_array)
    Simthread.result = np.repeat([np.nan], repeats=n)
    inputs = list()
    for ith in range(n):
        inputs.append({'para': para_array[ith], 'loc': ith})
    push_works(inputs)
    while np.isnan(np.min(Simthread.result)):
        time.sleep(1)
    batch_ans = Simthread.result.copy()
    Simthread.reset_results()
    return batch_ans


if __name__ == '__main__':
    t_start = time.time()

    # unittest for single simulation controls
    # fsim = Fullsim(sumo_config, sumo_rd_net_url, veh_logs_train, need_gui=False)
    sumo_config = './conf/aofeng.sumocfg'
    sumo_rd_net_url = './conf/xuancheng1116_6.net.xml'
    sumo_target_logs_csv = './conf/veh_stop_truth.csv'
    # outputs
    best_model_path = 'model/aofeng/18_epoch_model.pt'

    n_actions = 4
    n_observations = 7
    engine = Validation(sumo_config, sumo_rd_net_url,
                             sumo_target_logs_csv, best_model_path, n_observations, n_actions,
                             label='eng0')
    for ith in range(1):
        ans = engine.run_with(np.array([2, 2, 20]))
        print(str(ith) + " exp: " + str(ans))

    # # unittest for multiple simulation controls
    # for i_core in range(5):
    #     create_threads()
    # batch_para = np.array([[2, 2, 20], [2, 2, 20], [2, 2, 20], [2, 2, 20], [2, 2, 20]])
    # # np.repeat([[2, 2, 20]], repeats=10, axis=0)
    # results = batch(batch_para)
    # end_threads()

    # test for pso

    # 根据cpu核心数确定
    # for i_core in range(1):
    #     create_threads()
    #
    # # bounds
    # # [tau minGap maxSpd]
    # lower = np.array([0.5, 0.5, 20])
    # upper = np.array([2, 2, 40])
    # bounds = (lower, upper)
    # # max_bound = 5.12 * np.ones(2)
    # # min_bound = - max_bound
    # # bounds = (min_bound, max_bound)
    #
    # # Initialize swarm
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    #
    # # Call instance of PSO with bounds argument
    # optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=3, options=options, bounds=bounds)
    #
    # # Perform optimization
    # cost, pos = optimizer.optimize(batch, iters=2)
    # # 历史解
    # pos_his = open("calibration/record/pos_history.pickle", 'wb')
    # pickle.dump(optimizer.pos_history, pos_his)
    # pos_his.close()
    # # 每代最优目标函数
    # cos_his = open("calibration/record/cost_history.pickle", 'wb')
    # pickle.dump(optimizer.cost_history, cos_his)
    # cos_his.close()
    #
    # # Initialize mesher with objected function
    # # m = Mesher(func=batch)
    # # 会讲历史解重算一次，非必要请勿使用
    # # ph3d_his = open('calibration/record/pos_history.pickle', 'wb')
    # # ph3d = m.compute_history_3d(optimizer.pos_history)
    # # pickle.dump(ph3d, ph3d_his)
    # # ph3d_his.close()
    #
    # end_threads()
    #
    # duration = time.time() - t_start
    # print("used time: " + str(duration) + " s.")
    #
    # print("DEBUG")
    # https: // pyswarms.readthedocs.io / en / latest / examples / tutorials / visualization.html  # Plotting-in-2-D-space
