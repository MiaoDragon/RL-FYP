import random
# memory
class ReplayMemory(object):
    # cyclic buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, reward, next_state, term):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, reward, next_state, term)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
