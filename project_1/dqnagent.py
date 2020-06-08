import torch
import torch.nn.functional as F
from .Memory import ReplayBuffer
from .model import QNetwork
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    
    def __init__(self,states,actions,learn_sch,gamma,seed=0,use_ddqn=False):
        self.state_size = states
        self.action_size = actions
        
        self.gamma = gamma
        self.qnet_local = QNetwork(states,actions).to(device)
        self.qnet_target = QNetwork(states,actions).to(device)
        self.learn_schedule = learn_sch
        self.lcount=0
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(),lr =5e-4)
        self.seed = random.seed(seed)
        self.ddqn = use_ddqn
        self.memory= ReplayBuffer()
        
        
    def e_greedy(self,state,eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device) 
        self.qnet_local.eval() # evaluation mode ( no backprop. )
        with torch.no_grad():
            action_logits = self.qnet_local(state_tensor)
        self.qnet_local.train() # evaluation mode off
        
        if random.random()>eps:
            
            return np.argmax(action_logits.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def store_exp(self,state,action,reward,next_state,done):
        
        self.memory.add_experience(state,action,reward,next_state,done)
        
        self.lcount = (self.lcount+1)%self.learn_schedule
        
        if self.lcount == 0:
            if len(self.memory)>self.memory.batch_size :
                exp_sample = self.memory.sample(device)
                self.learn(exp_sample)
    
    def learn(self,sample):
        
        states,actions,rewards,next_states,dones = sample
        
        # DDQN
        if self.ddqn:
            tar_temp  = self.qnet_local(next_states).detach().argmax(1).unsqueeze(1)
            targets_q = self.qnet_target(next_states).gather(1,tar_temp)
        
        # Vanilla
        else:
            
            targets_q  = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        
        targets = rewards + (self.gamma*targets_q*(1-dones))
        
        expected = self.qnet_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(expected, targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnet_local, self.qnet_target, 1e-3)
        
        
    def soft_update(self,net_local,net_target,tau):
        
        for target_param, local_param in zip(net_target.parameters(), net_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
          