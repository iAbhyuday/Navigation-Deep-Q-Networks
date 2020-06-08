import torch
import torch.nn.functional as F
class QNetwork(torch.nn.Module):
    
    def __init__(self,state_size,num_actions,seed=0):
        super(QNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 =  torch.nn.Linear(state_size,128)
        self.fc2 = torch.nn.Linear(128,64)
        self.fc3 = torch.nn.Linear(64,num_actions)
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)
        
        