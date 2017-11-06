from DQN.Improver import Improver
from DQN.Evaluator import Evaluator
#from DQN.DQN import DQN
from DQN.DQNcartpole import DQN
from Env.Environment import Environment
from DQN.ReplayMemory import ReplayMemory
import torch.nn as nn
import torch.optim as optim
from Env.gymEnv import myGym
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Manager, Event
from multiprocessing.managers import SyncManager
# hyperparameters
MEMORY_SIZE = 100

imp_net = DQN()
#eval_net = DQN()
#eval_target_net = DQN()
#eval_net.setOptimizer(optim.RMSprop(eval_net.parameters(), lr=LEARNING_RATE,
#                                    momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
#                                    eps=MIN_SQUARED_GRAD))
imp_net.share_memory()
#eval_net.share_memory()
#eval_target_net.share_memory()
#env = Environment()
#env = myGym()


# populate memory
# let improver populate first
SyncManager.register('ReplayMemory', ReplayMemory,exposed=['getCapacity', 'push',
                                                        'sample', '__len__'])
manager = SyncManager()
#memory = ReplayMemory(MEMORY_SIZE)
manager.start()
lst = manager.list()
memory = manager.ReplayMemory(MEMORY_SIZE, lst)
shared = manager.dict({'memory':memory, 'SENT_FLAG':True, 'weights':None})

#print('create improver, evaluator...')
#time.sleep(1)
improver = Improver(imp_net, shared, myGym())
# improver is executed by the main process
evaluator = Evaluator(shared)

threads = []
#improver.start()
evaluator.start()  # fork & exec the evaluator
improver.run()
threads.append(evaluator)

for t in threads:
    t.join()
