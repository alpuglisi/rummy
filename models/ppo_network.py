import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RummyActorCritic(nn.Module):
    # Layers are attributes named shared_fc1..N so a state_dict saved by the
    # original fixed 256x2 network loads unchanged into (hidden_size=256, num_layers=2).
    def __init__(self, obs_dim=318, action_dim=105, hidden_size=512, num_layers=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        in_dim = obs_dim
        for i in range(1, num_layers + 1):
            setattr(self, f"shared_fc{i}", nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size

        head = hidden_size // 2
        self.actor_fc = nn.Linear(hidden_size, head)
        self.actor_logits = nn.Linear(head, action_dim)

        self.critic_fc = nn.Linear(hidden_size, head)
        self.critic_value = nn.Linear(head, 1)

    def forward(self, state, action_mask=None):
        x = state
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"shared_fc{i}")(x))

        c = F.relu(self.critic_fc(x))
        value = self.critic_value(c)

        a = F.relu(self.actor_fc(x))
        logits = self.actor_logits(a)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        return logits, value

    def get_action(self, state, action_mask):
        logits, value = self.forward(state, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @classmethod
    def from_state_dict(cls, state_dict):
        """Rebuild the network with whatever architecture the checkpoint was saved with."""
        hidden_size, obs_dim = state_dict["shared_fc1.weight"].shape
        num_layers = sum(1 for k in state_dict if k.startswith("shared_fc") and k.endswith(".weight"))
        action_dim = state_dict["actor_logits.weight"].shape[0]
        model = cls(obs_dim, action_dim, hidden_size, num_layers)
        model.load_state_dict(state_dict)
        return model
