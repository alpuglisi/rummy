import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env.vectorized_env import CHANNELS_END, EVENT_DIM, HISTORY, HISTORY_LEN, OBS_DIM, SCALARS

# Auxiliary head output layout (see trainer.aux_losses for the targets).
AUX_SLICES = {
    "opponent": slice(0, 52),          # cards in the opponent's hand (BCE)
    "next_discard": slice(52, 104),    # the opponent's next discard (52-way CE)
    "layoff": slice(104, 156),         # table cards whose meld the opponent can extend (BCE)
    "takeable": slice(156, 208),       # own hand cards the opponent could take if discarded (BCE)
    "discard_value": slice(208, 260),  # points / 50 the opponent scores at once from each such discard (MSE)
    "flags": slice(260, 263),          # opponent holds a meld, can go out next turn, deck completes a meld (BCE)
    "hand_points": slice(263, 264),    # opponent hand points / 100 (MSE)
    "turns_left": slice(264, 265),     # turns until the round ends / 40 (MSE)
    "goes_out": slice(265, 268),       # who ends the round: me, opponent, nobody (3-way CE)
}
AUX_OUT = 268


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class RummyActorCritic(nn.Module):
    """Shared trunk with actor, critic and auxiliary heads.

    structured=True (current): the six 52-card channels are also read as a
    6 x 4 x 13 suit-by-rank grid by a small convolution, so neighbouring
    ranks and matching ranks across suits are built in; the turn history is
    read by a small transformer over its HISTORY_LEN events; both feed the
    residual trunk together with the raw channels and scalars.
    structured=False keeps the earlier flat architectures loadable:
    residual=True is the plain input projection + residual blocks,
    residual=False the original MLP (layers named shared_fc1..N).
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=105, hidden_size=512, num_layers=4, residual=True,
                 structured=True, history_dim=64, conv_channels=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.residual = residual
        self.structured = structured
        self.has_aux = True   # False when loaded from a checkpoint saved without the auxiliary head
        if structured:
            if obs_dim != OBS_DIM:
                raise ValueError(f"the structured network reads the full {OBS_DIM}-wide observation")
            self.card_conv = nn.Sequential(
                nn.Conv2d(6, conv_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU())
            self.hist_embed = nn.Linear(EVENT_DIM, history_dim)
            self.hist_pos = nn.Parameter(torch.zeros(HISTORY_LEN, history_dim))
            layer = nn.TransformerEncoderLayer(history_dim, nhead=4, dim_feedforward=2 * history_dim,
                                               dropout=0.0, batch_first=True, norm_first=True)
            self.hist_enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            in_dim = CHANNELS_END + conv_channels * 4 * 13 + 2 * history_dim + SCALARS
            self.input = nn.Linear(in_dim, hidden_size)
            self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_layers))
            self.final_norm = nn.LayerNorm(hidden_size)
        elif residual:
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

        # Auxiliary heads: hidden facts the engine knows (opponent's hand and
        # what it can do, what the deck holds, how the round ends). Trained
        # only during PPO so the trunk learns to infer them from public play;
        # unused at inference except the belief-weighted search.
        if structured:
            self.aux_head = nn.Linear(hidden_size, AUX_OUT)
        else:
            self.aux_opponent = nn.Linear(hidden_size, 52)

    def features(self, state):
        channels = state[:, :CHANNELS_END]
        grid = channels.reshape(-1, 6, 4, 13)                       # card = suit * 13 + rank
        conv = self.card_conv(grid).flatten(1)
        hist = state[:, HISTORY].reshape(-1, HISTORY_LEN, EVENT_DIM)
        tokens = self.hist_embed(hist) + self.hist_pos             # empty slots become a learned "no event" token
        enc = self.hist_enc(tokens)
        pooled = torch.cat([enc.mean(dim=1), enc[:, -1]], dim=-1)  # summary + most recent event
        return torch.cat([channels, conv, pooled, state[:, -SCALARS:]], dim=-1)

    def trunk(self, state):
        if self.structured:
            x = F.relu(self.input(self.features(state)))
            for block in self.blocks:
                x = block(x)
            return self.final_norm(x)
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

    def aux(self, x):
        """Auxiliary head outputs as a dict of raw (pre-activation) tensors."""
        if self.structured:
            out = self.aux_head(x)
            return {name: out[:, sl] for name, sl in AUX_SLICES.items()}
        return {"opponent": self.aux_opponent(x)}

    def forward_with_aux(self, state, action_mask=None):
        x = self.trunk(state)
        logits, value = self.heads(x, action_mask)
        return logits, value, self.aux(x)

    def get_action(self, state, action_mask):
        logits, value = self.forward(state, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @classmethod
    def from_state_dict(cls, state_dict):
        """Rebuild the network with whatever architecture the checkpoint was saved with."""
        action_dim = state_dict["actor_logits.weight"].shape[0]
        if "card_conv.0.weight" in state_dict:
            hidden_size = state_dict["input.weight"].shape[0]
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            model = cls(OBS_DIM, action_dim, hidden_size, num_layers, structured=True,
                        history_dim=state_dict["hist_embed.weight"].shape[0],
                        conv_channels=state_dict["card_conv.0.weight"].shape[0])
            aux_prefix = "aux_head."
        elif "input.weight" in state_dict:
            hidden_size, obs_dim = state_dict["input.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=True, structured=False)
            aux_prefix = "aux_opponent."
        else:
            hidden_size, obs_dim = state_dict["shared_fc1.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("shared_fc") and k.endswith(".weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=False, structured=False)
            aux_prefix = "aux_opponent."
        # Checkpoints saved before the auxiliary heads exist without them; they are not needed to play.
        result = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not k.startswith(aux_prefix)]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {unexpected_missing}, unexpected {result.unexpected_keys}")
        model.has_aux = not any(k.startswith(aux_prefix) for k in result.missing_keys)
        return model
