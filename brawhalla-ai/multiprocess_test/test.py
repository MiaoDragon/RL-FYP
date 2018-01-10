from multiprocessing.managers import SyncManager
import torch.nn as nn
import time
from DQN.DQNcartpole import DQN
from Env.gymEnv import myGym
from DQN.CartPoleDQN import CartPoleDQN
from DQN.ReplayMemory import ReplayMemory

if __name__ == '__main__':
    demonet = DQN()
    #manager = SyncManager()
    #manager.start()
    memory = ReplayMemory(10000)
    #for i in range(memory.capacity):
    #    memory.push(torch.FloatTensor(1, 3, 40, 80))
    shared = dict({'memory':memory, 'SENT_FLAG': True, 'weights': None})
    p = CartPoleDQN(DQN(),shared,myGym())
    p.run()
