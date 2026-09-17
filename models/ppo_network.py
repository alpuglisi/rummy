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
# Target groups in the order the trainer's per-game dropout mask uses.
AUX_GROUPS = list(AUX_SLICES)
CARD_HEADS = ["opponent", "next_discard", "layoff", "takeable", "discard_value"]   # one value per card
GLOBAL_AUX = 3 + 1 + 1 + 3   # flags, hand_points, turns_left, goes_out

CARD_FEATURES = 6 + 13 + 4   # six channels + rank one-hot + suit one-hot


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


def card_rank_suit():
    cards = torch.arange(52)
    return torch.cat([F.one_hot(cards % 13, 13), F.one_hot(cards // 13, 4)], dim=1).float()   # [52, 17]


class RummyActorCritic(nn.Module):
    """Shared trunk with actor, critic and auxiliary heads. Three layouts:

    arch="tokens" (current): every card is a token carrying its six channel
    bits and its rank and suit; the round's history events are tokens too;
    one global token carries the scalars. A transformer attends over all of
    them. Per-card decisions come from the card tokens (discard logits, "take
    the pile down to this card" logits, and the per-card auxiliary targets);
    the deck-draw logit, the value and the global auxiliary targets come from
    the global token through a residual trunk.
    arch="structured": the earlier flat model with a convolution over the
    card grid and a transformer over the history block.
    arch="flat": the original MLP / residual-MLP models (obs_dim 320 or less).
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=105, hidden_size=512, num_layers=4, residual=True,
                 arch="tokens", history_dim=64, conv_channels=32, history_len=HISTORY_LEN,
                 token_dim=128, token_layers=4, token_heads=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.residual = residual
        self.arch = arch
        self.history_len = history_len
        self.has_aux = True   # False when loaded from a checkpoint saved without the auxiliary heads
        head = hidden_size // 2
        if arch == "tokens":
            if obs_dim != OBS_DIM:
                raise ValueError(f"the token network reads the full {OBS_DIM}-wide observation")
            self.token_dim = token_dim
            self.register_buffer("rank_suit", card_rank_suit(), persistent=False)
            self.card_embed = nn.Linear(CARD_FEATURES, token_dim)
            self.card_pos = nn.Parameter(torch.zeros(52, token_dim))
            self.event_embed = nn.Linear(EVENT_DIM, token_dim)
            self.event_pos = nn.Parameter(torch.zeros(history_len, token_dim))
            self.global_embed = nn.Linear(SCALARS, token_dim)
            self.global_pos = nn.Parameter(torch.zeros(1, token_dim))
            layer = nn.TransformerEncoderLayer(token_dim, nhead=token_heads, dim_feedforward=2 * token_dim,
                                               dropout=0.0, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=token_layers, enable_nested_tensor=False)
            self.encoder_norm = nn.LayerNorm(token_dim)
            # Per-card outputs: discard logit, take-pile logit, then the five per-card aux targets.
            self.card_head = nn.Linear(token_dim, 2 + len(CARD_HEADS))
            self.input = nn.Linear(2 * token_dim, hidden_size)   # global token + mean of card tokens
            self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_layers))
            self.final_norm = nn.LayerNorm(hidden_size)
            self.deck_logit = nn.Linear(hidden_size, 1)
            self.global_aux = nn.Linear(hidden_size, GLOBAL_AUX)
        elif arch == "structured":
            self.card_conv = nn.Sequential(
                nn.Conv2d(6, conv_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU())
            self.hist_embed = nn.Linear(EVENT_DIM, history_dim)
            self.hist_pos = nn.Parameter(torch.zeros(history_len, history_dim))
            layer = nn.TransformerEncoderLayer(history_dim, nhead=4, dim_feedforward=2 * history_dim,
                                               dropout=0.0, batch_first=True, norm_first=True)
            self.hist_enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            in_dim = CHANNELS_END + conv_channels * 4 * 13 + 2 * history_dim + SCALARS
            self.input = nn.Linear(in_dim, hidden_size)
            self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_layers))
            self.final_norm = nn.LayerNorm(hidden_size)
            self.actor_fc = nn.Linear(hidden_size, head)
            self.actor_logits = nn.Linear(head, action_dim)
            self.aux_head = nn.Linear(hidden_size, AUX_OUT)
        else:
            if residual:
                self.input = nn.Linear(obs_dim, hidden_size)
                self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_layers))
                self.final_norm = nn.LayerNorm(hidden_size)
            else:
                in_dim = obs_dim
                for i in range(1, num_layers + 1):
                    setattr(self, f"shared_fc{i}", nn.Linear(in_dim, hidden_size))
                    in_dim = hidden_size
            self.actor_fc = nn.Linear(hidden_size, head)
            self.actor_logits = nn.Linear(head, action_dim)
            self.aux_opponent = nn.Linear(hidden_size, 52)

        self.critic_fc = nn.Linear(hidden_size, head)
        self.critic_value = nn.Linear(head, 1)

    # ---- token architecture -------------------------------------------------
    def encode_tokens(self, state):
        b = state.shape[0]
        channels = state[:, :CHANNELS_END].reshape(b, 6, 52).transpose(1, 2)          # [B, 52, 6]
        card_feats = torch.cat([channels, self.rank_suit.expand(b, 52, 17)], dim=-1)   # [B, 52, 23]
        cards = self.card_embed(card_feats) + self.card_pos
        events = state[:, HISTORY].reshape(b, HISTORY_LEN, EVENT_DIM)[:, -self.history_len:]
        events = self.event_embed(events) + self.event_pos
        glob = self.global_embed(state[:, -SCALARS:]).unsqueeze(1) + self.global_pos
        tokens = torch.cat([glob, cards, events], dim=1)
        out = self.encoder_norm(self.encoder(tokens))
        return out[:, 0], out[:, 1:53]                                                  # global, cards

    @staticmethod
    def pile_slots(state):
        """For each card in the pile, its action slot (1 + depth index) from the depth channel."""
        depth = state[:, 2 * 52:3 * 52]
        n = (depth > 0).sum(dim=1, keepdim=True).clamp(min=1)
        slot = torch.round(depth * n).long()            # depth is (i+1)/n for the card at index i
        return slot, depth > 0

    def tokens_forward(self, state, action_mask=None):
        glob, cards = self.encode_tokens(state)
        per_card = self.card_head(cards)                                       # [B, 52, 7]
        x = F.relu(self.input(torch.cat([glob, cards.mean(dim=1)], dim=-1)))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        b = state.shape[0]
        logits = torch.zeros(b, 105, device=state.device, dtype=per_card.dtype)
        logits[:, 0:1] = self.deck_logit(x)
        slot, in_pile = self.pile_slots(state)
        take = per_card[:, :, 1] * in_pile                                     # only pile cards can be taken
        logits.scatter_add_(1, torch.where(in_pile, slot, torch.zeros_like(slot)), take)
        logits[:, 53:] = per_card[:, :, 0]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        c = F.relu(self.critic_fc(x))
        value = self.critic_value(c)
        return logits, value, x, per_card

    def tokens_aux(self, x, per_card):
        aux = {name: per_card[:, :, 2 + i] for i, name in enumerate(CARD_HEADS)}
        g = self.global_aux(x)
        aux["flags"] = g[:, 0:3]
        aux["hand_points"] = g[:, 3:4]
        aux["turns_left"] = g[:, 4:5]
        aux["goes_out"] = g[:, 5:8]
        return aux

    # ---- structured / flat architectures -------------------------------------
    def features(self, state):
        channels = state[:, :CHANNELS_END]
        grid = channels.reshape(-1, 6, 4, 13)                       # card = suit * 13 + rank
        conv = self.card_conv(grid).flatten(1)
        hist = state[:, HISTORY].reshape(-1, HISTORY_LEN, EVENT_DIM)[:, -self.history_len:]
        tokens = self.hist_embed(hist) + self.hist_pos             # empty slots become a learned "no event" token
        enc = self.hist_enc(tokens)
        pooled = torch.cat([enc.mean(dim=1), enc[:, -1]], dim=-1)  # summary + most recent event
        return torch.cat([channels, conv, pooled, state[:, -SCALARS:]], dim=-1)

    def trunk(self, state):
        if self.arch == "structured":
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

    def aux(self, x):
        """Auxiliary head outputs as a dict of raw (pre-activation) tensors."""
        if self.arch == "structured":
            out = self.aux_head(x)
            return {name: out[:, sl] for name, sl in AUX_SLICES.items()}
        return {"opponent": self.aux_opponent(x)}

    # ---- common interface ----------------------------------------------------
    def forward(self, state, action_mask=None):
        if self.arch == "tokens":
            logits, value, _, _ = self.tokens_forward(state, action_mask)
            return logits, value
        return self.heads(self.trunk(state), action_mask)

    def forward_with_aux(self, state, action_mask=None):
        if self.arch == "tokens":
            logits, value, x, per_card = self.tokens_forward(state, action_mask)
            return logits, value, self.tokens_aux(x, per_card)
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
        if "card_embed.weight" in state_dict:
            hidden_size = state_dict["input.weight"].shape[0]
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            token_layers = len({k.split(".")[2] for k in state_dict if k.startswith("encoder.layers.")})
            model = cls(OBS_DIM, 105, hidden_size, num_layers, arch="tokens",
                        token_dim=state_dict["card_embed.weight"].shape[0], token_layers=token_layers,
                        history_len=state_dict["event_pos"].shape[0])
            aux_prefix = ("card_head.", "global_aux.")
        elif "card_conv.0.weight" in state_dict:
            action_dim = state_dict["actor_logits.weight"].shape[0]
            hidden_size = state_dict["input.weight"].shape[0]
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            model = cls(OBS_DIM, action_dim, hidden_size, num_layers, arch="structured",
                        history_dim=state_dict["hist_embed.weight"].shape[0],
                        conv_channels=state_dict["card_conv.0.weight"].shape[0],
                        history_len=state_dict["hist_pos"].shape[0])
            aux_prefix = ("aux_head.",)
        elif "input.weight" in state_dict:
            action_dim = state_dict["actor_logits.weight"].shape[0]
            hidden_size, obs_dim = state_dict["input.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".fc1.weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=True, arch="flat")
            aux_prefix = ("aux_opponent.",)
        else:
            action_dim = state_dict["actor_logits.weight"].shape[0]
            hidden_size, obs_dim = state_dict["shared_fc1.weight"].shape
            num_layers = sum(1 for k in state_dict if k.startswith("shared_fc") and k.endswith(".weight"))
            model = cls(obs_dim, action_dim, hidden_size, num_layers, residual=False, arch="flat")
            aux_prefix = ("aux_opponent.",)
        # Checkpoints saved before the auxiliary heads exist without them; they are not needed to play.
        result = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not k.startswith(aux_prefix)]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {unexpected_missing}, unexpected {result.unexpected_keys}")
        model.has_aux = not any(k.startswith(aux_prefix) for k in result.missing_keys)
        return model
