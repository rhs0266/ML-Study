from collections import namedtuple
from random import sample

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
                        
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        elem = Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(elem)
        else:
            self.memory[self.position] = elem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
