from multiprocessing.managers import SyncManager
import torch.nn as nn
import time
from DQN.DQNcartpole import DQN
from Env.gymEnv import myGym
from DQN.MyProcess import MyProcess
from DQN.MainProcess import MainProcess
from DQN.ReplayMemory import ReplayMemory

if __name__ == '__main__':
    demonet = DQN()
    manager = SyncManager()
    manager.start()
    memory = ReplayMemory(100)
    #for i in range(memory.capacity):
    #    memory.push(torch.FloatTensor(1, 3, 40, 80))
    shared = manager.dict({'inputs':memory, 'SENT_FLAG': True, 'weights': None})

    pm = MainProcess(shared, demonet, myGym())
    p = MyProcess(shared)
    #p = multiprocessing.Process(target=iteration, args=(inputs,))
    p.start()
    pm.run()
    p.join()
