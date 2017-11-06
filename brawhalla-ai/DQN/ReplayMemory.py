import random
# memory
class ReplayMemory(object):
    # cyclic buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def getCapacity(self):
        return self.capacity

    def push(self, exp_tuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = exp_tuple
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
