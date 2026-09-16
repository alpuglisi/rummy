
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RummyActorCritic(nn.Module):
    def __init__(self, obs_dim=159, action_dim=105):
        super(RummyActorCritic, self).__init__()
        
        self.shared_fc1 = nn.Linear(obs_dim, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        
        self.actor_fc = nn.Linear(256, 128)
        self.actor_logits = nn.Linear(128, action_dim)
        
        self.critic_fc = nn.Linear(256, 128)
        self.critic_value = nn.Linear(128, 1)

    def forward(self, state, action_mask=None):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        c = F.relu(self.critic_fc(x))
        value = self.critic_value(c)
        
        a = F.relu(self.actor_fc(x))
        logits = self.actor_logits(a)
        
        if action_mask is not None:
            # Masking FP16 underflow fix
            huge_negative = torch.tensor(torch.finfo(logits.dtype).min, device=logits.device)
            logits = torch.where(action_mask, logits, huge_negative)
            
        return logits, value

    def get_action(self, state, action_mask):
        logits, value = self.forward(state, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value
