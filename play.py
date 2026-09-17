"""Play Rummy against the trained model.

    python play.py [checkpoint] [--worlds 64 --actions 6 --seed N --no-search]

Deals are random on every launch; pass --seed to replay a particular game
(the seed used is printed at start).

Without a checkpoint argument the game uses checkpoints/best.pth, which the
trainer updates whenever the live policy beats the previous best head-to-head.

Draw phase: click the deck, or click a discard-pile card to take the pile from
that card up. Only cards you can legally take are highlighted: the deepest card
must be playable at once, either as a new meld with your hand or laid off onto
a meld on the table. Then click cards in your hand to select them: press M (or
the button) to lay the selection down as a meld, or select one card and click
a meld on the table to lay it off, as often as you like. Finally select a
single card and press D (or the button) to discard it and end your turn. You
must always keep one card to discard; the only way to go out is to discard
your last card. The computer melds and lays off automatically, as it was
trained to.

Scoring: Ace 1, 2-10 face value, J/Q/K 10, earned as cards go on the table.
Going out (or the deck running out) ends the round: each player's leftover
cards are subtracted from their own score, the table is cleared and a new
round is dealt. The game is won by the first player to reach 500 at the end
of a round. Keys: A toggles the advisor (the search's expected round outcome
for each of your legal moves, assuming you lay down everything you can: a
ranked list appears top right and the best move is framed in green), N
starts a new game, Q quits.
"""
import argparse
import os
import sys

import numpy as np
import pygame
import torch

import rummy_engine
from config import PPOConfig
from evaluate import ModelPolicy, load_model
from search import SearchPolicy

WIDTH, HEIGHT = 1280, 800
CARD_W, CARD_H = 80, 116
TABLE_W, TABLE_H = 56, 81          # melds on the table are drawn smaller
SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
GREEN = (30, 100, 60)
HIGHLIGHT = (255, 215, 0)
TEXT = (245, 245, 245)
DIM = (170, 190, 175)
BEST = (130, 235, 150)


def card_name(card):
    rank, suit = RANKS[card % 13], SUITS[card // 13]
    return f"{rank[0].upper() if rank in ('jack', 'queen', 'king', 'ace') else rank}{'♣♦♥♠'[card // 13]}"


class Assets:
    def __init__(self, folder="assets/cards"):
        self.faces = {}
        self.small_faces = {}
        for card in range(52):
            rank, suit = RANKS[card % 13], SUITS[card // 13]
            path = os.path.join(folder, f"{rank}_of_{suit}2.png")   # court illustrations where available
            if not os.path.exists(path):
                path = os.path.join(folder, f"{rank}_of_{suit}.png")
            img = pygame.image.load(path).convert_alpha()
            self.faces[card] = pygame.transform.smoothscale(img, (CARD_W, CARD_H))
            self.small_faces[card] = pygame.transform.smoothscale(img, (TABLE_W, TABLE_H))
        self.back = pygame.Surface((CARD_W, CARD_H), pygame.SRCALPHA)
        pygame.draw.rect(self.back, (250, 250, 250), self.back.get_rect(), border_radius=8)
        pygame.draw.rect(self.back, (40, 60, 140), self.back.get_rect().inflate(-10, -10), border_radius=6)
        for y in range(12, CARD_H - 12, 10):
            pygame.draw.line(self.back, (90, 110, 190), (12, y), (CARD_W - 12, y + 6), 2)


class Game:
    def __init__(self, model, device, args):
        self.env = rummy_engine.RummyEnv(args.seed)
        self.human = 1
        self.computer = ModelPolicy(model, device) if args.no_search else SearchPolicy(
            model, device, worlds=args.worlds, max_actions=args.actions, seed=args.seed)
        self.advisor = SearchPolicy(model, device, worlds=args.worlds, max_actions=6, seed=args.seed + 1)
        self.show_advice = False
        self.advice = {}
        self.selected = set()
        self.log = []
        self.result = None
        self.env.set_manual_meld(self.human)
        self.new_game()

    # --- state helpers -------------------------------------------------------
    def new_game(self):
        self.env.reset()
        self.round_start = (0.0, 0.0)
        self.log = [f"New game to {self.env.get_target_score()}. You are player 1; you draw first."]
        self.result = None
        self.advice = {}
        self.selected.clear()
        self.refresh_advice()

    def obs(self):
        return self.env.get_state()

    def phase_is_discard(self):
        return self.obs()[-3] == 1.0

    def pile(self):
        o = self.obs()
        depth = o[104:156]
        cards = np.flatnonzero(depth > 0)
        return cards[np.argsort(depth[cards])]          # oldest first, top of pile last

    def hand(self):
        # The observation is from the player to move; on the computer's turn the
        # human's cards are the engine's "opponent hand" for that view.
        if self.env.get_current_player() == self.human:
            return np.flatnonzero(self.obs()[:52] == 1.0)
        return np.flatnonzero(self.env.get_opponent_hand())

    def board(self):
        return np.flatnonzero(self.obs()[156:208] == 1.0)

    def table(self):
        return self.env.get_table()

    def opponent_size(self):
        return int(self.env.get_opponent_hand().sum()) if self.env.get_current_player() == self.human \
            else int(self.obs()[:52].sum())

    def legal(self):
        return self.env.get_legal_actions().astype(bool)

    def refresh_advice(self):
        self.advice = {}
        if self.show_advice and not self.env.is_done() and self.env.get_current_player() == self.human:
            self.advice = self.advisor.evaluate_actions([self.env])[0]

    def describe_action(self, action):
        if action == 0:
            return "Draw from the deck"
        if action <= 52:
            pile = self.pile()
            taken = len(pile) - (action - 1)
            return f"Take the pile down to {card_name(int(pile[action - 1]))} ({taken} card{'s' if taken > 1 else ''})"
        return f"Discard {card_name(action - 53)}"

    def advice_rows(self):
        """Advisor suggestions, best first: (action, description, expected margin, gap to best)."""
        if not self.advice:
            return []
        ranked = sorted(self.advice.items(), key=lambda kv: -kv[1][0])
        best = ranked[0][1][0]
        return [(a, self.describe_action(a), v, best - v) for a, (v, _) in ranked]

    def best_action(self):
        rows = self.advice_rows()
        return rows[0][0] if rows else None

    # --- moves ---------------------------------------------------------------
    def apply(self, action, who):
        pile_before = self.pile()
        if action == 0:
            text = "drew from the deck"
        elif action <= 52:
            taken = pile_before[action - 1:]
            text = f"took the pile down to {card_name(int(pile_before[action - 1]))} ({len(taken)} cards)"
        else:
            text = f"discarded {card_name(action - 53)}"
        round_no = self.env.get_round()
        deck_empty = self.obs()[-2] >= 1.0
        reward, done = self.env.step(int(action))
        self.log.append(f"{who} {text}.")
        if self.env.round_ended():
            self.end_round(round_no, "Deck ran out" if deck_empty else f"{who} went out")
        if done:
            self.finish()
        self.refresh_advice()

    def end_round(self, round_no, how):
        s1, s2 = self.env.get_score(1), self.env.get_score(2)
        gain_you, gain_cpu = s1 - self.round_start[0], s2 - self.round_start[1]
        self.round_start = (s1, s2)
        self.selected.clear()
        self.log.append(f"Round {round_no} over ({how}): you {gain_you:+.0f}, computer {gain_cpu:+.0f}.  "
                        f"Score {s1:.0f} - {s2:.0f}.")

    def finish(self):
        s1, s2 = self.env.get_score(1), self.env.get_score(2)
        penalised = self.env.get_penalised()
        if penalised == self.human:
            outcome = "Computer wins: you broke the pile-draw obligation."
        elif penalised:
            outcome = "You win: the computer broke the pile-draw obligation."
        elif s1 == s2:
            outcome = "Draw."
        else:
            outcome = "You win!" if s1 > s2 else "Computer wins."
        self.result = f"{outcome}  Final score: you {s1:.0f} - computer {s2:.0f}.  Press N for a new game."
        self.log.append(self.result)

    def toggle_select(self, card):
        self.selected.symmetric_difference_update({int(card)})

    def obligation_ok(self, cards):
        """The card taken from the pile must be played before anything else."""
        required = self.env.get_required_card()
        return required == -1 or required in cards

    def can_meld(self):
        hand = set(int(c) for c in self.hand())
        return (3 <= len(self.selected) < len(hand) and self.selected <= hand
                and self.obligation_ok(self.selected) and self.env.is_valid_meld(sorted(self.selected)))

    def can_discard(self):
        return len(self.selected) == 1 and self.legal()[53 + next(iter(self.selected))]

    def can_lay_off(self, meld_index):
        hand = set(int(c) for c in self.hand())
        if len(self.selected) != 1 or len(hand) < 2 or not self.selected <= hand \
                or not self.obligation_ok(self.selected):
            return False
        return self.env.can_lay_off(next(iter(self.selected)), self.table()[meld_index])

    def lay_off_selected(self, meld_index):
        if not self.can_lay_off(meld_index):
            return
        card = next(iter(self.selected))
        try:
            reward, done = self.env.lay_off(card, meld_index)
        except ValueError as err:
            self.log.append(f"Can't lay that off: {err}.")
            return
        self.selected.clear()
        self.log.append(f"You laid off {card_name(card)}.")
        if done:
            self.finish(reward, "You")
        self.refresh_advice()

    def meld_selected(self):
        if not self.can_meld():
            return
        cards = sorted(self.selected)
        try:
            reward, done = self.env.meld(cards)
        except ValueError as err:
            self.log.append(f"Can't meld that: {err}.")
            return
        self.selected.clear()
        self.log.append("You melded " + " ".join(card_name(c) for c in cards) + ".")
        if done:
            self.finish(reward, "You")
        self.refresh_advice()

    def discard_selected(self):
        if not self.can_discard():
            return
        card = next(iter(self.selected))
        self.selected.clear()
        self.apply(53 + card, "You")

    def computer_turn(self):
        while not self.env.is_done() and self.env.get_current_player() == self.computer_seat():
            if hasattr(self.computer, "act_envs"):
                action = int(self.computer.act_envs([self.env])[0])
            else:
                action = int(self.computer.act(self.obs()[None], self.legal()[None])[0])
            self.apply(action, "Computer")

    def computer_seat(self):
        return 2 if self.human == 1 else 1


# --- rendering ------------------------------------------------------------------
class View:
    def __init__(self, screen, assets):
        self.screen = screen
        self.assets = assets
        self.font = pygame.font.SysFont("dejavusans", 18)
        self.small = pygame.font.SysFont("dejavusans", 14)
        self.big = pygame.font.SysFont("dejavusans", 26, bold=True)
        self.hit = []   # (rect, action)

    def text(self, s, x, y, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(s, True, color), (x, y))

    def card(self, card, x, y, face=True, highlight=False, label=None, best=False):
        img = self.assets.faces[card] if face else self.assets.back
        self.screen.blit(img, (x, y))
        if highlight:
            pygame.draw.rect(self.screen, BEST if best else HIGHLIGHT,
                             (x - 2, y - 2, CARD_W + 4, CARD_H + 4), 3, border_radius=8)
        if label is not None:
            tag = self.small.render(label, True, (20, 20, 20))
            box = tag.get_rect(center=(x + CARD_W // 2, y - 12)).inflate(8, 4)
            pygame.draw.rect(self.screen, BEST if best else HIGHLIGHT, box, border_radius=4)
            self.screen.blit(tag, tag.get_rect(center=box.center))
        return pygame.Rect(x, y, CARD_W, CARD_H)

    def advice_label(self, game, action):
        """Short tag drawn above a card the advisor evaluated, e.g. 'BEST +14' or '+9'."""
        if action not in game.advice:
            return None, False
        value = game.advice[action][0]
        best = action == game.best_action()
        return (f"BEST {value:+.0f}" if best else f"{value:+.0f}"), best

    def advisor_panel(self, game, discard):
        """Ranked, plain-language list of the advisor's suggestions (top right)."""
        x, y = 700, 40
        rows = game.advice_rows()
        self.text("Advisor: expected points advantage (this round, plus the value of the position after it)",
                  x, y, self.small, DIM)
        y += 20
        if not rows:
            self.text("thinking..." if game.show_advice else "", x, y, self.small, DIM)
            return
        for rank, (action, text, value, gap) in enumerate(rows[:6], 1):
            color = BEST if rank == 1 else TEXT
            note = "best move" if rank == 1 else f"{gap:.0f} pts worse"
            self.text(f"{rank}. {text}", x, y, self.small, color)
            self.text(f"{value:+.0f}", x + 330, y, self.small, color)
            self.text(note, x + 380, y, self.small, DIM if rank > 1 else BEST)
            y += 18
        hint = ("Numbers assume you lay down every meld and lay-off you can before discarding."
                if discard else "Only the model's likeliest moves are rated; unlisted moves were not considered.")
        self.text(hint, x, y + 2, self.small, DIM)

    def draw(self, game):
        g, s = game, self.screen
        s.fill(GREEN)
        self.hit = []
        human_turn = not g.env.is_done() and g.env.get_current_player() == g.human
        legal = g.legal() if human_turn else np.zeros(105, dtype=bool)
        discard = g.phase_is_discard() if human_turn else False

        # Opponent hand (top): always face down, even cards taken from the pile
        opp_n = g.opponent_size()
        self.text(f"Computer  -  {opp_n} cards, score {g.env.get_score(g.computer_seat()):.0f}", 20, 12, self.big)
        self.text(f"Round {g.env.get_round()}   first to {g.env.get_target_score()}",
                  WIDTH - 260, 18, self.font, DIM)
        x = 20
        for _ in range(opp_n):
            self.card(0, x, 45, face=False); x += 26

        # Table: each meld is a group; a single selected hand card can be laid
        # off by clicking a group it extends.
        table = g.table()
        self.text(f"Table ({len(table)} melds)", 20, 175, self.font, DIM)
        x, y = 20, 198
        for idx, meld in enumerate(table):
            width = TABLE_W + 18 * (len(meld) - 1)
            if x + width > WIDTH - 20:
                x, y = 20, y + 54
            rect = pygame.Rect(x, y, width, TABLE_H)
            for k, card in enumerate(meld):
                s.blit(self.assets.small_faces[int(card)], (x + 18 * k, y))
            if human_turn and discard and g.can_lay_off(idx):
                pygame.draw.rect(s, HIGHLIGHT, rect.inflate(6, 6), 3, border_radius=6)
                self.hit.append((rect, ("layoff", idx)))
            x += width + 16

        # Deck + pile (middle)
        deck_left = 52 - int(round(float(g.obs()[-2]) * 52))
        y = 335
        deck_ok = human_turn and not discard and legal[0]
        rect = self.card(0, 20, y, face=False, highlight=deck_ok, best=deck_ok and g.best_action() == 0)
        self.text(f"Deck: {deck_left}", 20, y + CARD_H + 6, self.small, DIM)
        if human_turn and not discard and legal[0]:
            self.hit.append((rect, ("action", 0)))
        pile = g.pile()
        x = 140
        self.text("Discard pile (oldest -> newest)", x, y + CARD_H + 6, self.small, DIM)
        step = min(60, max(28, (WIDTH - 180) // max(1, len(pile))))
        for i, card in enumerate(pile):
            action = 1 + i
            ok = human_turn and not discard and legal[action]
            label, best = self.advice_label(g, action) if ok else (None, False)
            rect = self.card(int(card), x, y, highlight=ok, label=label, best=best)
            if ok:
                self.hit.append((rect, ("action", action)))
            x += step
        if human_turn and not discard and 0 in g.advice:
            label, best = self.advice_label(g, 0)
            self.text(f"deck: {label}", 20, y + CARD_H + 24, self.small, BEST if best else HIGHLIGHT)

        # Human hand (bottom)
        hand = g.hand()
        self.text(f"You  -  {len(hand)} cards, score {g.env.get_score(g.human):.0f}", 20, 490, self.big)
        x = 20
        step = min(90, max(30, (WIDTH - 40) // max(1, len(hand))))
        for card in hand:
            action = 53 + int(card)
            ok = human_turn and discard and legal[action]
            chosen = int(card) in g.selected
            label, best = self.advice_label(g, action) if ok else (None, False)
            rect = self.card(int(card), x, 540 if chosen else 560, highlight=ok, label=label, best=best)
            if chosen:
                pygame.draw.rect(s, (80, 160, 255), rect.inflate(6, 6), 3, border_radius=8)
            if human_turn and discard:
                self.hit.append((rect, ("select", int(card))))
            x += step
        if g.show_advice and human_turn:
            self.advisor_panel(g, discard)
        if human_turn and discard:
            self.button("Meld selected  (M)", WIDTH - 470, 690, g.can_meld(), ("meld", None))
            self.button("Discard selected  (D)", WIDTH - 240, 690, g.can_discard(), ("discard", None))

        # Status + log
        if g.result:
            status = g.result
        elif human_turn:
            if not discard:
                status = "Your turn: click the deck or a pile card you can take."
            elif len(g.selected) == 1:
                status = "Your turn: click a highlighted table meld to lay off, or D to discard."
            else:
                status = "Your turn: select cards to meld (M), or one card to lay off / discard (D)."
        else:
            status = "Computer is thinking..."
        self.text(status, 20, 690, self.font, HIGHLIGHT)
        for i, line in enumerate(g.log[-4:]):
            self.text(line, 20, 715 + 18 * i, self.small, DIM)
        self.text("A: advisor " + ("on" if g.show_advice else "off") + "   N: new game   Q: quit",
                  WIDTH - 330, HEIGHT - 24, self.small, DIM)
        pygame.display.flip()

    def button(self, label, x, y, enabled, value):
        color = HIGHLIGHT if enabled else (90, 120, 100)
        rect = pygame.Rect(x, y, 210, 30)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        tag = self.font.render(label, True, (20, 20, 20) if enabled else DIM)
        self.screen.blit(tag, tag.get_rect(center=rect.center))
        if enabled:
            self.hit.append((rect, value))

    def click(self, pos):
        for rect, value in reversed(self.hit):   # topmost card wins
            if rect.collidepoint(pos):
                return value
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=PPOConfig.best_checkpoint,
                        help="model to play against (default: the trainer's best checkpoint)")
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None,
                        help="random by default; set to replay the same deals and search worlds")
    parser.add_argument("--no-search", action="store_true", help="computer plays the plain policy")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint} (train first, or pass a .pth file)")
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "little") % (2**31 - 1)
    print(f"Seed {args.seed} (pass --seed {args.seed} to replay these deals)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Playing against {args.checkpoint}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rummy vs the model")
    view = View(screen, Assets())
    game = Game(model, device, args)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_n:
                    game.new_game()
                elif event.key == pygame.K_a:
                    game.show_advice = not game.show_advice
                    game.refresh_advice()
                elif event.key == pygame.K_m:
                    game.meld_selected()
                elif event.key == pygame.K_d:
                    game.discard_selected()
                    view.draw(game)
                    game.computer_turn()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = view.click(event.pos)
                if hit is not None:
                    kind, value = hit
                    if kind == "action":
                        game.apply(value, "You")
                    elif kind == "select":
                        game.toggle_select(value)
                    elif kind == "meld":
                        game.meld_selected()
                    elif kind == "layoff":
                        game.lay_off_selected(value)
                    elif kind == "discard":
                        game.discard_selected()
                        view.draw(game)
                        game.computer_turn()
        view.draw(game)
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
