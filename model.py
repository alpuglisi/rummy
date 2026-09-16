import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RummyActorCritic(nn.Module):
    def __init__(self, obs_dim=159, action_dim=105):
        super(RummyActorCritic, self).__init__()
        
        # Shared Feature Extractor
        # Maps the 159-dim state (52 Hand + 52 Discard Presence + 52 Discard Depth + 3 Meta)
        self.shared_fc1 = nn.Linear(obs_dim, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        
        # Actor Head (Outputs probability distribution over 105 actions)
        self.actor_fc = nn.Linear(256, 128)
        self.actor_logits = nn.Linear(128, action_dim)
        
        # Critic Head (Outputs expected value of the current state)
        self.critic_fc = nn.Linear(256, 128)
        self.critic_value = nn.Linear(128, 1)

    def forward(self, state, action_mask=None):
        """
        state: Tensor of shape [batch_size, 159]
        action_mask: Boolean Tensor of shape [batch_size, 105] (True for legal actions)
        """
        # Shared backbone
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Critic prediction
        c = F.relu(self.critic_fc(x))
        value = self.critic_value(c)
        
        # Actor prediction (Raw logits)
        a = F.relu(self.actor_fc(x))
        logits = self.actor_logits(a)
        
        # Apply the C++ Action Mask
        if action_mask is not None:
            # Replace illegal action logits with a massive negative number
            # so their probability becomes 0 after softmax
            huge_negative = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits = torch.where(action_mask, logits, huge_negative)
            
        return logits, value

    def get_action(self, state, action_mask):
        """
        Samples an action for the environment step.
        """
        logits, value = self.forward(state, action_mask)
        
        # Create a categorical distribution over the legal actions
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Sample an action
        action = dist.sample()
        
        # Return action, the log probability of that action, and the state value
        return action, dist.log_prob(action), value
