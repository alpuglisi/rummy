import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class RummyActorCritic(nn.Module):
    """Shared trunk with actor and critic heads.

    residual=True: input projection, num_layers pre-norm residual blocks, final
    LayerNorm. residual=False: the original plain MLP, whose layers are named
    shared_fc1..N so checkpoints from the earlier fixed networks load unchanged.
    """

    def __init__(self, obs_dim=318, action_dim=105, hidden_size=512, num_layers=4, residual=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.residual = residual
        self.has_aux = True   # False when loaded from a checkpoint saved without the auxiliary head
        if residual:
            self.input = nn.Linear(obs_dim, hidden_size)
            self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_layers))
            self.final_norm = nn.LayerNorm(hidden_size)
        else:
            in_dim = obs_dim
            for i in range(1, num_layers + 1):
                setattr(self, f"shared_fc{i}", nn.Linear(in_dim, hidden_size))
                in_dim = hidden_size

        head = hidden_size // 2
        self.actor_fc = nn.Linear(hidden_size, head)
        self.actor_logits = nn.Linear(head, action_dim)

        self.critic_fc = nn.Linear(hidden_size, head)
        self.critic_value = nn.Linear(head, 1)

        # Auxiliary head: which cards the opponent holds. Trained against the
        # engine's true state so the trunk learns to infer hidden information
        # from public play; unused at inference.
        self.aux_opponent = nn.Linear(hidden_size, 52)

    def trunk(self, state):
        if self.residual:
            x = F.relu(self.input(state))
            for block in self.blocks:
                x = block(x)
            return self.final_norm(x)
        x = state
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"shared_fc{i}")(x))
        return x

    def heads(self, x, action_mask=None):
        c = F.relu(self.critic_fc(x))
        value = self.critic_value(c)

        a = F.relu(self.actor_fc(x))
        logits = self.actor_logits(a)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        return logits, value

    def forward(self, state, action_mask=None):
        return self.heads(self.trunk(state), action_mask)

    def forward_with_aux(self, state, action_mask=None):
        x = self.trunk(state)
        logits, value = self.heads(x, action_mask)
        return logits, value, self.aux_opponent(x)

    def get_action(self, state, action_mask):
        logits, value = self.forward(state, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @classmethod
    def from_state_dict(cls, state_dict):
        """Rebuild the network with whatever architecture the checkpoint was saved with."""
        action_dim = state_dict["actor_logits.weight"].shape[0]
        if "input.weight" in state_dict:
            hidden_size, obs_dim = state_dict["input.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=True)
        else:
            hidden_size, obs_dim = state_dict["shared_fc1.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("shared_fc") and k.endswith(".weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=False)
        # Checkpoints saved before the auxiliary head exist without it; it is not needed to play.
        result = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not k.startswith("aux_opponent.")]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {unexpected_missing}, unexpected {result.unexpected_keys}")
        model.has_aux = not any(k.startswith("aux_opponent.") for k in result.missing_keys)
        return model
