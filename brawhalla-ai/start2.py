from multiprocessing.managers import SyncManager
import multiprocessing
import torch.nn as nn
import time
from DQN.DQNcartpole import DQN
#from Env.Environment import Environment
from Env.gymEnv_V2 import myGym
#from Env.gymEnv import myGym
from DQN.Improver_Q_Learning2 import Improver
from DQN.Evaluator_Q_Learning import Evaluator
from DQN.ReplayMemory import ReplayMemory
import os
os.system("taskset -p 0xff %d" % os.getpid())  #https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy

if __name__ == '__main__':
    # hyperparameters
    MEMORY_SIZE = 20000
    #MEMORY_SIZE = 5
    imp_net = DQN()
    # populate memory
    # let improver populate first
    manager = SyncManager()
    manager.start()
    memory = ReplayMemory(MEMORY_SIZE)
    s = multiprocessing.Semaphore(1)
    #memory = multiprocessing.Queue(MEMORY_SIZE)
    memory = manager.list()
    shared = manager.dict({'SENT_FLAG':True, 'weights':None})
    #shared = manager.dict({'memory':memory, 'SENT_FLAG':True, 'weights':None})
    #improver = Improver(imp_net, shared, myGym(), s)
    improver = Improver(imp_net, MEMORY_SIZE, memory, shared, myGym(), s)
    # improver is executed by the main process
    evaluator = Evaluator(memory, shared, s)

    evaluator.start()  # fork & exec the evaluator
    improver.run()
    evaluator.join()
