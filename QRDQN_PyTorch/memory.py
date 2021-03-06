import random
import numpy as np
from collections import deque

# random
class Memory :

    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
        self.memory_counter = 0

    def add(self, sample) :
        self.memory.append(sample)
        self.memory_counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)