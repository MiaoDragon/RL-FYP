from multiprocessing.managers import SyncManager
import multiprocessing
import torch.nn as nn
import time
#from DQN.NNcartpole import DQN
from DQN.DQN import DQN
#from Env.Environment import Environment
#from Env.gymEnv_V3 import myGym
#from Env.gymEnv import myGym
from DQN.Improver_Q_Learning_Braw import Improver
from DQN.Evaluator_Q_Learning_3_Braw import Evaluator
#from DQN.Evaluator_Dense_Q_Learning3 import Evaluator
from DQN.ReplayMemory import ReplayMemory
import os
def wait(T):
    for i in list(range(T))[::-1]:
        print(i+1)
        time.sleep(1)

if __name__ == '__main__':
    # hyperparameters
    wait(5)
    multiprocessing.set_start_method('spawn')
    os.system("taskset -p 0xff %d" % os.getpid())  #https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
    MEMORY_SIZE = 10000
    imp_net = DQN()
    # populate memory
    # let improver populate first
    manager = SyncManager()
    manager.start()
    #s = multiprocessing.Semaphore(1)
    #memory = multiprocessing.Queue(MEMORY_SIZE)
    memory = manager.list()
    shared = manager.dict({'SENT_FLAG':True, 'weights':None})
    #shared = manager.dict({'memory':memory, 'SENT_FLAG':True, 'weights':None})
    #improver = Improver(imp_net, shared, myGym(), s)
    improver = Improver(imp_net, MEMORY_SIZE, memory, shared)
    # improver is executed by the main process
    evaluator = Evaluator(memory, shared)

    evaluator.start()  # fork & exec the evaluator
    improver.run()
    evaluator.join()
