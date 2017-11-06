from multiprocessing.managers import SyncManager
import torch.nn as nn
import time
from DQN.DQNcartpole import DQN
#from Env.Environment import Environment
from Env.gymEnv import myGym
from DQN.Improver import Improver
from DQN.Evaluator import Evaluator
from DQN.ReplayMemory import ReplayMemory
if __name__ == '__main__':
    # hyperparameters
    MEMORY_SIZE = 500
    imp_net = DQN()
    # populate memory
    # let improver populate first
    manager = SyncManager()
    manager.start()
    memory = ReplayMemory(MEMORY_SIZE)
    shared = manager.dict({'memory':memory, 'SENT_FLAG':True, 'weights':None})
    improver = Improver(imp_net, shared, myGym())
    # improver is executed by the main process
    evaluator = Evaluator(shared)

    evaluator.start()  # fork & exec the evaluator
    improver.run()
    evaluator.join()
