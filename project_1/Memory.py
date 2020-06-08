from collections import deque,namedtuple
import torch
import random
import numpy as np
class ReplayBuffer:
    def __init__(self,seed=0,max_size=int(2e5),batch_size=128,prioritized=False):
        self.max_size = max_size
        self.isPrioritized = prioritized
        self.batch_size = batch_size
        self.exp=namedtuple('Experience',['state','action','reward','next_state','done'])
        self.memory = deque(maxlen = self.max_size)
        self.seed = random.seed(seed)
    def add_experience(self,state,action,reward,next_state,done):
        
        self.memory.append(self.exp(state,action,reward,next_state,done))
        
        
    def sample(self,device):
        experiences = random.sample(self.memory,self.batch_size)
        
        states  = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions  = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards  = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states  = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones  = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        return len(self.memory)
        
