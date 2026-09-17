#include "rummy_env.h"

#include <string>

// --- Helper Functions ---

int RummyEnv::get_point_value(int card) const {
    int rank = get_rank(card);
    if (rank == 0) return 1;   // Ace
    if (rank >= 9) return 10;  // 10, J, Q, K
    return rank + 1;           // 2-9 at face value
}

void RummyEnv::find_all_possible_melds(const Hand& hand, MeldList& out) const {
    out.n = 0;

    // 1. Sets (3 or 4 of a kind)
    for (int rank = 0; rank < 13; rank++) {
        SmallMeld current_set;
        for (int card : hand) {
            if (get_rank(card) == rank) current_set.push(card);
        }
        if (current_set.size() >= 3) out.push(current_set);
    }

    // 2. Runs (3+ consecutive in suit)
    for (int suit = 0; suit < 4; suit++) {
        SmallMeld suit_cards;
        for (int card : hand) {
            if (get_suit(card) == suit) suit_cards.push(card);
        }
        std::sort(suit_cards.begin(), suit_cards.end());

        for (int i = 0; i < suit_cards.size(); i++) {
            SmallMeld current_run;
            current_run.push(suit_cards[i]);
            for (int j = i + 1; j < suit_cards.size(); j++) {
                if (get_rank(suit_cards[j]) == get_rank(current_run.back()) + 1) {
                    current_run.push(suit_cards[j]);
                    if (current_run.size() >= 3) out.push(current_run);
                } else {
                    break;
                }
            }
        }
    }
}

SmallMeld RummyEnv::find_largest_meld_with_card(const Hand& hand, int card) const {
    MeldList melds;
    find_all_possible_melds(hand, melds);
    SmallMeld best;
    for (int i = 0; i < melds.size(); i++) {
        const SmallMeld& m = melds[i];
        if (m.contains(card) && m.size() > best.size()) best = m;
    }
    return best;
}

bool RummyEnv::can_take_pile(int player, size_t index) const {
    const int card = state.discard_pile[index];
    Hand combined = get_hand(player);
    for (size_t i = index; i < state.discard_pile.size(); i++) combined.push(state.discard_pile[i]);
    SmallMeld m = find_largest_meld_with_card(combined, card);
    // A fresh meld needs a 4th card left in hand (a larger meld is trimmed to
    // leave a discard, see the discard phase); otherwise the card may extend
    // a meld already on the table.
    return (!m.empty() && combined.size() >= 4) || (find_lay_off(card) >= 0 && combined.size() > 1);
}

int RummyEnv::immediate_points(const Hand& hand, int card) const {
    Hand combined = hand;
    combined.push(card);
    SmallMeld m = find_largest_meld_with_card(combined, card);
    if (!m.empty() && combined.size() >= 4) {
        int pts = 0;
        for (int c : m) pts += get_point_value(c);
        return pts;
    }
    if (find_lay_off(card) >= 0 && combined.size() > 1) return get_point_value(card);
    return 0;
}

// Table melds are Python-visible std::vectors; this converts one back
// for placing or saving on the stack.
static SmallMeld to_small_meld(const std::vector<int>& cards) {
    if (cards.size() > static_cast<size_t>(MAX_MELD_CARDS)) throw std::logic_error("meld larger than a suit");
    SmallMeld m;
    for (int c : cards) m.push(c);
    return m;
}

bool RummyEnv::could_go_out(int player, const Hand& extra) const {
    // Simulate in place and restore: auto_meld only touches card locations,
    // the table and the scores (never the mask cache), and copying the whole
    // engine per simulated draw was the cost of aux_targets(). The existing
    // table melds are saved on the stack (lay offs may extend and re-sort
    // them) and restored with assign, which fits in their kept capacity, so
    // the only heap traffic is a fresh meld laid during the simulation.
    RummyEnv& self = const_cast<RummyEnv&>(*this);
    const auto saved_locations = state.card_locations;
    const size_t saved_n = state.table.size();
    if (saved_n > static_cast<size_t>(MAX_TABLE_MELDS)) throw std::logic_error("more table melds than the deck allows");
    std::array<SmallMeld, MAX_TABLE_MELDS> saved_table;
    for (size_t i = 0; i < saved_n; i++) saved_table[i] = to_small_meld(state.table[i]);
    const float saved_p1 = state.p1_score, saved_p2 = state.p2_score;
    for (int c : extra) self.state.card_locations[c] = static_cast<int8_t>(player);
    self.auto_meld(player);
    const bool out = hand_count(player) <= 1;
    self.state.card_locations = saved_locations;
    self.state.table.resize(saved_n);
    for (size_t i = 0; i < saved_n; i++) self.state.table[i].assign(saved_table[i].begin(), saved_table[i].end());
    self.state.p1_score = saved_p1;
    self.state.p2_score = saved_p2;
    return out;
}

std::vector<std::array<int, 4>> RummyEnv::get_history() const {
    std::vector<std::array<int, 4>> out;
    out.reserve(state.history.size());
    for (const Event& e : state.history) out.push_back({e.player, e.kind, e.card, e.count});
    return out;
}

void RummyEnv::aux_targets(float* out) const {
    std::fill(out, out + AUX_SPACE_SIZE, 0.0f);
    const int me = state.current_player;
    const int opp = (me == 1) ? 2 : 1;
    const Hand my_hand = get_hand(me);
    const Hand opp_hand = get_hand(opp);
    float* opp_cards = out;
    float* layoff_targets = out + DECK_SIZE;
    float* takeable = out + 2 * DECK_SIZE;
    float* discard_value = out + 3 * DECK_SIZE;
    float* scalars = out + 4 * DECK_SIZE;

    for (int c : opp_hand) opp_cards[c] = 1.0f;
    for (const Meld& m : state.table) {
        bool extendable = false;
        for (int c : opp_hand) if (can_lay_off(c, m)) { extendable = true; break; }
        if (extendable) for (int c : m) layoff_targets[c] = 1.0f;
    }
    for (int c : my_hand) {
        const int pts = immediate_points(opp_hand, c);
        if (pts > 0) {
            takeable[c] = 1.0f;
            discard_value[c] = std::min(1.0f, static_cast<float>(pts) / 50.0f);
        }
    }
    MeldList opp_melds;
    find_all_possible_melds(opp_hand, opp_melds);
    scalars[0] = opp_melds.empty() ? 0.0f : 1.0f;
    int opp_points = 0;
    for (int c : opp_hand) opp_points += get_point_value(c);
    scalars[1] = static_cast<float>(opp_points) / 100.0f;

    // Can the opponent go out on their next turn? Try the deck's next card
    // and every pile take that would be legal for them.
    bool go_out = false;
    if (deck_index < DECK_SIZE) {
        Hand next;
        next.push(deck_order[deck_index]);
        go_out = could_go_out(opp, next);
    }
    for (size_t i = 0; i < state.discard_pile.size() && !go_out; i++) {
        if (can_take_pile(opp, i)) {
            Hand taken;
            for (size_t j = i; j < state.discard_pile.size(); j++) taken.push(state.discard_pile[j]);
            go_out = could_go_out(opp, taken);
        }
    }
    scalars[2] = go_out ? 1.0f : 0.0f;

    // Would one of the next three deck cards complete a meld or lay-off for me?
    for (int k = 0; k < 3 && deck_index + k < DECK_SIZE; k++) {
        const int d = deck_order[deck_index + k];
        if (immediate_points(my_hand, d) > 0) { scalars[3] = 1.0f; break; }
    }
}

py::array_t<float> RummyEnv::get_aux_targets() const {
    py::array_t<float> out(AUX_SPACE_SIZE);
    aux_targets(out.mutable_data());
    return out;
}

Hand RummyEnv::get_hand(int player) const {
    Hand hand;
    for (int i = 0; i < DECK_SIZE; i++) {
        if (state.card_locations[i] == player) hand.push(i);
    }
    return hand;
}

int RummyEnv::hand_count(int player) const {
    int n = 0;
    for (int i = 0; i < DECK_SIZE; i++) n += state.card_locations[i] == player;
    return n;
}

void RummyEnv::auto_meld(int player) {
    MeldList possible_melds;
    bool found_meld = true;
    while (found_meld) {
        found_meld = false;
        find_all_possible_melds(get_hand(player), possible_melds);

        if (!possible_melds.empty()) {
            // Greedily pick the highest scoring meld (first found wins ties)
            int best = 0;
            int best_score = 0;

            for (int i = 0; i < possible_melds.size(); i++) {
                int current_score = 0;
                for (int c : possible_melds[i]) current_score += get_point_value(c);
                if (current_score > best_score) {
                    best_score = current_score;
                    best = i;
                }
            }

            place_meld(player, possible_melds[best]);
            found_meld = true; // Check again to see if remaining cards form new melds
        }
    }

    // Lay off any remaining card onto a meld already on the table.
    bool laid = true;
    while (laid) {
        laid = false;
        for (int c : get_hand(player)) {
            int idx = find_lay_off(c);
            if (idx >= 0) {
                lay_off_card(player, c, idx);
                laid = true;
                break;
            }
        }
    }
}

void RummyEnv::place_meld(int player, const SmallMeld& m) {
    SmallMeld sorted = m;
    std::sort(sorted.begin(), sorted.end());
    for (int c : sorted) {
        state.card_locations[c] = 4;
        if (player == 1) state.p1_score += get_point_value(c);
        else state.p2_score += get_point_value(c);
    }
    state.table.emplace_back(sorted.begin(), sorted.end());
}

void RummyEnv::lay_off_card(int player, int card, int index) {
    state.card_locations[card] = 4;
    if (player == 1) state.p1_score += get_point_value(card);
    else state.p2_score += get_point_value(card);
    Meld& m = state.table[index];
    m.push_back(card);
    std::sort(m.begin(), m.end());
}

int RummyEnv::find_lay_off(int card) const {
    for (size_t i = 0; i < state.table.size(); i++) {
        if (can_lay_off(card, state.table[i])) return static_cast<int>(i);
    }
    return -1;
}

bool RummyEnv::can_lay_off(int card, const Meld& meld) const {
    if (card < 0 || card >= DECK_SIZE || meld.empty()) return false;
    if (std::find(meld.begin(), meld.end(), card) != meld.end()) return false;
    bool is_set = true;
    for (int c : meld) if (get_rank(c) != get_rank(meld[0])) is_set = false;
    if (is_set) return meld.size() < 4 && get_rank(card) == get_rank(meld[0]);
    // Run: same suit, one rank below the lowest or above the highest.
    if (get_suit(card) != get_suit(meld[0])) return false;
    int lo = 13, hi = -1;
    for (int c : meld) {
        lo = std::min(lo, get_rank(c));
        hi = std::max(hi, get_rank(c));
    }
    return get_rank(card) == lo - 1 || get_rank(card) == hi + 1;
}

// --- Environment Logic ---

RummyEnv::RummyEnv(uint32_t seed, int target_score, int max_rounds, int hand_size)
    : rng(seed), deck_index(0), target_score(target_score), max_rounds(max_rounds), hand_size(hand_size) {
    if (hand_size < 1 || 2 * hand_size + 1 > DECK_SIZE) throw std::invalid_argument("hand_size out of range");
    for (int i = 0; i < DECK_SIZE; i++) deck_order[i] = static_cast<int8_t>(i);
    reset();
}

void RummyEnv::deal_initial_hands() {
    for (int i = 0; i < hand_size; i++) {
        state.card_locations[deck_order[deck_index++]] = 1;
        state.card_locations[deck_order[deck_index++]] = 2;
    }
}

void RummyEnv::reset() {
    state.current_player = 1;
    state.is_terminal = false;
    state.p1_score = 0.0f;
    state.p2_score = 0.0f;
    state.round_number = 1;
    state.penalised_player = 0;
    state.last_round_out = 0;
    start_round();
    state.round_just_ended = false;
    mask_valid = false;
}

void RummyEnv::start_round() {
    std::shuffle(deck_order.begin(), deck_order.end(), rng);
    deck_index = 0;

    state.card_locations.fill(0);
    state.discard_pile.clear();
    state.table.clear();
    state.history.clear();
    state.publicly_known.fill(0);
    state.required_meld_card = -1;
    state.turn_phase_is_discard = false;
    state.cached_required_meld.clear();
    state.round_start_p1 = state.p1_score;
    state.round_start_p2 = state.p2_score;

    deal_initial_hands();

    int first_discard = deck_order[deck_index++];
    state.card_locations[first_discard] = 3;
    state.discard_pile.push_back(first_discard);
    mask_valid = false;
}

float RummyEnv::end_round(int acting_player) {
    state.last_round_out = hand_count(acting_player) == 0 ? acting_player : 0;
    int hand_value[3] = {0, 0, 0};
    for (int i = 0; i < DECK_SIZE; i++) {
        const int loc = state.card_locations[i];
        if (loc == 1 || loc == 2) hand_value[loc] += get_point_value(i);
    }
    state.p1_score -= static_cast<float>(hand_value[1]);
    state.p2_score -= static_cast<float>(hand_value[2]);

    const float own_gain = (acting_player == 1) ? state.p1_score - state.round_start_p1
                                                : state.p2_score - state.round_start_p2;
    const float opp_gain = (acting_player == 1) ? state.p2_score - state.round_start_p2
                                                : state.p1_score - state.round_start_p1;
    float reward = own_gain - opp_gain;

    state.round_just_ended = true;
    if (state.p1_score >= target_score || state.p2_score >= target_score || state.round_number >= max_rounds) {
        state.is_terminal = true;
        if (state.p1_score != state.p2_score) {
            const int winner = (state.p1_score > state.p2_score) ? 1 : 2;
            reward += (winner == acting_player) ? 100.0f : -100.0f;
        }
    } else {
        state.round_number++;
        state.current_player = (acting_player == 1) ? 2 : 1;
        start_round();
    }
    mask_valid = false;
    return reward;
}

void RummyEnv::observe(float* out) const {
    std::fill(out, out + OBS_SPACE_SIZE, 0.0f);

    for (int i = 0; i < DECK_SIZE; i++) {
        // Channel 1: Hand
        if (state.card_locations[i] == state.current_player) {
            out[i] = 1.0f;
        }
    }

    // Channel 2 & 3: Discard Presence and Depth/Order
    for (size_t i = 0; i < state.discard_pile.size(); i++) {
        int card = state.discard_pile[i];
        out[DECK_SIZE + card] = 1.0f;
        out[DECK_SIZE * 2 + card] = static_cast<float>(i + 1) / state.discard_pile.size();
    }

    // Channel 4, 5 & 6: melded cards (dead for everyone), opponent's known hand
    // cards, and unseen cards (deck or opponent's unknown hand: anything the
    // current player has no information about and could still be drawn).
    const int opponent = (state.current_player == 1) ? 2 : 1;
    int own_hand_size = 0, opp_hand_size = 0;
    for (int i = 0; i < DECK_SIZE; i++) {
        if (state.card_locations[i] == 4) out[DECK_SIZE * 3 + i] = 1.0f;
        if (state.card_locations[i] == opponent) {
            opp_hand_size++;
            if (state.publicly_known[i]) out[DECK_SIZE * 4 + i] = 1.0f;
            else out[DECK_SIZE * 5 + i] = 1.0f;
        }
        if (state.card_locations[i] == 0) out[DECK_SIZE * 5 + i] = 1.0f;
        if (state.card_locations[i] == state.current_player) own_hand_size++;
    }

    // Turn history: the last HISTORY_LEN events, most recent in the last slot.
    const int base = DECK_SIZE * 6;
    const int n_events = static_cast<int>(state.history.size());
    const int first = std::max(0, n_events - HISTORY_LEN);
    for (int e = first; e < n_events; e++) {
        const Event& ev = state.history[e];
        float* slot = out + base + (HISTORY_LEN - (n_events - e)) * EVENT_DIM;
        slot[0] = (ev.player != state.current_player) ? 1.0f : 0.0f;
        slot[1 + ev.kind] = 1.0f;
        if (ev.card >= 0) {
            slot[4 + get_rank(ev.card)] = 1.0f;
            slot[4 + 13 + get_suit(ev.card)] = 1.0f;
        }
        slot[EVENT_DIM - 1] = static_cast<float>(ev.count) / 10.0f;
    }

    const float own_score = (state.current_player == 1) ? state.p1_score : state.p2_score;
    const float opp_score = (state.current_player == 1) ? state.p2_score : state.p1_score;
    out[OBS_SPACE_SIZE - 8] = own_score / static_cast<float>(target_score);
    out[OBS_SPACE_SIZE - 7] = opp_score / static_cast<float>(target_score);
    out[OBS_SPACE_SIZE - 6] = static_cast<float>(own_hand_size) / 26.0f;
    out[OBS_SPACE_SIZE - 5] = static_cast<float>(opp_hand_size) / 26.0f;
    out[OBS_SPACE_SIZE - 4] = (own_score - opp_score) / 100.0f;

    // Meta variables
    out[OBS_SPACE_SIZE - 3] = state.turn_phase_is_discard ? 1.0f : 0.0f;
    out[OBS_SPACE_SIZE - 2] = static_cast<float>(deck_index) / DECK_SIZE;
    out[OBS_SPACE_SIZE - 1] = (state.required_meld_card != -1) ? 1.0f : 0.0f;
}

void RummyEnv::compute_legal_mask() const {
    std::array<uint8_t, ACTION_SPACE_SIZE>& mask = mask_cache;
    mask.fill(0);
    if (state.is_terminal) {
        mask_valid = true;
        return;
    }

    if (!state.turn_phase_is_discard) {
        // Draw Actions
        if (deck_index < DECK_SIZE) mask[0] = 1; // Can draw deck
        // A pile draw is only legal when the deepest card taken can be played
        // immediately: melded with the hand plus everything above it in the
        // pile, or laid off onto a meld already on the table. Either way a
        // card must remain to discard, since going out requires a discard.
        for (size_t i = 0; i < state.discard_pile.size(); i++) {
            if (can_take_pile(state.current_player, i)) mask[1 + i] = 1;
        }
    } else {
        // Discard Actions
        SmallMeld consumed_cards;
        if (state.required_meld_card != -1) {
            const Hand hand = get_hand(state.current_player);
            SmallMeld m = find_largest_meld_with_card(hand, state.required_meld_card);
            if (!m.empty() && m.size() == hand.size()) {
                // Melding everything would leave nothing to discard: drop one
                // end card of the meld (never the required card) so the
                // player keeps a discard, or fall back to a lay-off when the
                // meld is only three cards.
                if (m.size() > 3) {
                    std::sort(m.begin(), m.end());
                    if (m[0] != state.required_meld_card) m.pop_front();
                    else m.pop_back();
                } else {
                    m.clear();
                }
            }
            state.cached_required_meld.assign(m.begin(), m.end());
            if (m.empty() && find_lay_off(state.required_meld_card) >= 0) {
                // The pile card extends a table meld instead: it is laid off
                // at discard time and cannot itself be discarded.
                consumed_cards.push(state.required_meld_card);
            } else {
                consumed_cards = m;
            }
        }

        bool has_legal_discard = false;
        for (int i = 0; i < DECK_SIZE; i++) {
            if (state.card_locations[i] == state.current_player) {
                if (!consumed_cards.contains(i)) {
                    mask[53 + i] = 1;
                    has_legal_discard = true;
                }
            }
        }

        // If the required meld consumes the entire hand leaving no discard, unmask the hand.
        // Per design: a player must always discard a final card, even if that means
        // discarding a meld card and breaking the meld (resolve_meld() then applies -50).
        if (!has_legal_discard) {
            for (int i = 0; i < DECK_SIZE; i++) {
                if (state.card_locations[i] == state.current_player) mask[53 + i] = 1;
            }
        }
    }
    mask_valid = true;
}

bool RummyEnv::is_legal_action(int action) const {
    if (action < 0 || action >= ACTION_SPACE_SIZE) return false;
    return legal_mask()[action] == 1;
}

bool RummyEnv::resolve_meld(int player, int discard_card) {
    if (state.cached_required_meld.empty()) {
        // No meld from hand: the pile card must extend a meld on the table.
        const int card = state.required_meld_card;
        if (card == -1 || discard_card == card) return false;
        int idx = find_lay_off(card);
        if (idx < 0) return false;
        lay_off_card(player, card, idx);
        return true;
    }

    // Discarding a card that's part of the required meld breaks the meld:
    // the obligation isn't satisfied, so this is a failure (-50 penalty
    // at the call site), not a success.
    if (std::find(state.cached_required_meld.begin(), state.cached_required_meld.end(), discard_card)
        != state.cached_required_meld.end()) {
        return false;
    }

    place_meld(player, to_small_meld(state.cached_required_meld));
    state.cached_required_meld.clear(); // Clear cache
    return true;
}

py::tuple RummyEnv::step(int action) {
    auto result = step_raw(action);
    return py::make_tuple(result.first, result.second);
}

std::pair<float, bool> RummyEnv::step_raw(int action) {
    if (state.is_terminal) {
        return {0.0f, true};
    }

    if (action < 0 || action >= ACTION_SPACE_SIZE) {
        throw std::invalid_argument("action out of range [0, ACTION_SPACE_SIZE)");
    }
    // Also leaves state.cached_required_meld set for resolve_meld below.
    if (!is_legal_action(action)) {
        throw std::invalid_argument("illegal action for the current state (check get_legal_actions())");
    }

    int acting_player = state.current_player;
    state.round_just_ended = false;

    if (!state.turn_phase_is_discard) {
        // --- DRAW PHASE ---
        if (action == 0) {
            int drawn = deck_order[deck_index++];
            state.card_locations[drawn] = acting_player;
            state.required_meld_card = -1;
            state.history.push_back({static_cast<int8_t>(acting_player), 0, -1, 1});
        } else {
            int pile_index = action - 1;
            state.required_meld_card = state.discard_pile[pile_index];
            state.history.push_back({static_cast<int8_t>(acting_player), 1,
                                     static_cast<int8_t>(state.discard_pile[pile_index]),
                                     static_cast<int8_t>(state.discard_pile.size() - pile_index)});

            for (size_t i = pile_index; i < state.discard_pile.size(); i++) {
                state.card_locations[state.discard_pile[i]] = acting_player;
                state.publicly_known[state.discard_pile[i]] = 1;
            }
            state.discard_pile.erase(state.discard_pile.begin() + pile_index, state.discard_pile.end());
        }
        state.turn_phase_is_discard = true;
        mask_valid = false;
        return {0.0f, false};
    } else {
        // --- DISCARD PHASE ---
        int discard_card = action - 53;

        // 1. Enforce required deep-draw meld. Pass the card actually being
        //    discarded -- resolve_meld() fails if it's a member of the
        //    required meld (discarding it breaks the meld obligation).
        if (state.required_meld_card != -1) {
            bool melded = resolve_meld(acting_player, discard_card);
            if (!melded) {
                state.is_terminal = true;
                state.penalised_player = acting_player;
                mask_valid = false;
                return {-50.0f, true};
            }
        }

        // 2. Execute discard
        state.card_locations[discard_card] = 3;
        state.publicly_known[discard_card] = 0;
        state.discard_pile.push_back(discard_card);
        state.required_meld_card = -1;
        state.history.push_back({static_cast<int8_t>(acting_player), 2, static_cast<int8_t>(discard_card), 1});

        // 3. Auto-meld remaining valid sets/runs to score points naturally
        if (acting_player != manual_meld_player) auto_meld(acting_player);

        // The round ends when the acting player discards their last card
        // (after melding / laying off the rest) or the deck runs out. The
        // game itself ends only once a player has reached the target score.
        if (hand_count(acting_player) == 0 || deck_index >= DECK_SIZE) {
            float reward = end_round(acting_player);
            mask_valid = false;
            return {reward, state.is_terminal};
        }

        state.turn_phase_is_discard = false;
        state.current_player = (acting_player == 1) ? 2 : 1;

        mask_valid = false;
        return {0.1f, false};
    }
}

void RummyEnv::restore_deck_prefix() {
    // The redealt tail [deck_index, 52) is the hidden pool; the prefix must
    // hold every other card exactly once, or the next start_round() would
    // shuffle a deck with duplicates. The prefix order does not matter
    // because the whole array is reshuffled before it is dealt again.
    int k = 0;
    for (int c = 0; c < DECK_SIZE; c++) {
        if (state.card_locations[c] == 0) continue;
        if (k >= deck_index) throw std::logic_error("redeal: more cards out of the deck than were drawn");
        deck_order[k++] = static_cast<int8_t>(c);
    }
    if (k != deck_index) throw std::logic_error("redeal: fewer cards out of the deck than were drawn");
}

void RummyEnv::randomize_hidden(uint32_t seed) {
    const int opponent = (state.current_player == 1) ? 2 : 1;
    std::vector<int> pool;
    int unknown_in_hand = 0;
    for (int c = 0; c < DECK_SIZE; c++) {
        if (state.card_locations[c] == opponent && !state.publicly_known[c]) {
            pool.push_back(c);
            unknown_in_hand++;
        }
    }
    for (int i = deck_index; i < DECK_SIZE; i++) pool.push_back(deck_order[i]);

    std::mt19937 gen(seed);
    std::shuffle(pool.begin(), pool.end(), gen);

    for (size_t k = 0; k < pool.size(); k++) {
        if (static_cast<int>(k) < unknown_in_hand) {
            state.card_locations[pool[k]] = static_cast<int8_t>(opponent);
        } else {
            state.card_locations[pool[k]] = 0;
            deck_order[deck_index + (k - unknown_in_hand)] = static_cast<int8_t>(pool[k]);
        }
    }
    restore_deck_prefix();
    rng.seed(seed ^ 0x9e3779b9u);
    mask_valid = false;
}

void RummyEnv::randomize_hidden_weighted(uint32_t seed, const float* weights) {
    const int opponent = (state.current_player == 1) ? 2 : 1;
    std::vector<int> pool;
    int unknown_in_hand = 0;
    for (int c = 0; c < DECK_SIZE; c++) {
        if (state.card_locations[c] == opponent && !state.publicly_known[c]) {
            pool.push_back(c);
            unknown_in_hand++;
        }
    }
    for (int i = deck_index; i < DECK_SIZE; i++) pool.push_back(deck_order[i]);

    // Weighted sampling without replacement (Efraimidis-Spirakis): each card
    // gets key u^(1/w); the k largest keys are the opponent's cards.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uni(std::numeric_limits<double>::min(), 1.0);
    std::vector<std::pair<double, int>> keyed;
    keyed.reserve(pool.size());
    for (int c : pool) {
        const double w = std::max(1e-4, std::min(1.0, static_cast<double>(weights[c])));
        keyed.emplace_back(std::pow(uni(gen), 1.0 / w), c);
    }
    std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> rest;
    for (size_t k = 0; k < keyed.size(); k++) {
        const int c = keyed[k].second;
        if (static_cast<int>(k) < unknown_in_hand) {
            state.card_locations[c] = static_cast<int8_t>(opponent);
        } else {
            state.card_locations[c] = 0;
            rest.push_back(c);
        }
    }
    std::shuffle(rest.begin(), rest.end(), gen);
    for (size_t k = 0; k < rest.size(); k++) deck_order[deck_index + k] = static_cast<int8_t>(rest[k]);
    restore_deck_prefix();
    rng.seed(seed ^ 0x9e3779b9u);
    mask_valid = false;
}

void RummyEnv::randomize_hidden_weighted_py(uint32_t seed,
                                            py::array_t<float, py::array::c_style | py::array::forcecast> weights) {
    if (weights.ndim() != 1 || weights.shape(0) != DECK_SIZE) {
        throw std::invalid_argument("weights must have one entry per card");
    }
    randomize_hidden_weighted(seed, weights.data());
}

bool RummyEnv::is_valid_meld(const std::vector<int>& cards) const {
    if (cards.size() < 3) return false;
    std::vector<int> sorted(cards);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return false;
    for (int c : sorted) if (c < 0 || c >= DECK_SIZE) return false;

    bool same_rank = true, same_suit = true;
    for (int c : sorted) {
        if (get_rank(c) != get_rank(sorted[0])) same_rank = false;
        if (get_suit(c) != get_suit(sorted[0])) same_suit = false;
    }
    if (same_rank) return sorted.size() <= 4;
    if (!same_suit) return false;
    for (size_t i = 1; i < sorted.size(); i++) {
        if (get_rank(sorted[i]) != get_rank(sorted[i - 1]) + 1) return false;
    }
    return true;
}

py::tuple RummyEnv::meld(const std::vector<int>& cards) {
    if (state.is_terminal) throw std::invalid_argument("game is over");
    if (!state.turn_phase_is_discard) throw std::invalid_argument("melds are laid down after drawing");
    for (int c : cards) {
        if (c < 0 || c >= DECK_SIZE || state.card_locations[c] != state.current_player) {
            throw std::invalid_argument("all melded cards must be in your hand");
        }
    }
    if (!is_valid_meld(cards)) throw std::invalid_argument("not a valid set or run");
    // The card taken from the pile must be played before anything else, so a
    // player cannot lay down other melds that break its only meld.
    if (state.required_meld_card != -1 &&
        std::find(cards.begin(), cards.end(), state.required_meld_card) == cards.end()) {
        throw std::invalid_argument("you must first play the card you took from the pile");
    }
    const int player = state.current_player;
    if (static_cast<int>(cards.size()) >= hand_count(player)) {
        throw std::invalid_argument("you must keep a card to discard");
    }

    place_meld(player, to_small_meld(cards));
    if (std::find(cards.begin(), cards.end(), state.required_meld_card) != cards.end()) {
        state.required_meld_card = -1;
        state.cached_required_meld.clear();
    }
    mask_valid = false;
    return py::make_tuple(0.0f, false);
}

py::tuple RummyEnv::lay_off(int card, int meld_index) {
    if (state.is_terminal) throw std::invalid_argument("game is over");
    if (!state.turn_phase_is_discard) throw std::invalid_argument("cards are laid off after drawing");
    if (card < 0 || card >= DECK_SIZE || state.card_locations[card] != state.current_player) {
        throw std::invalid_argument("the card must be in your hand");
    }
    if (meld_index < 0 || meld_index >= static_cast<int>(state.table.size())) {
        throw std::invalid_argument("no such meld on the table");
    }
    if (!can_lay_off(card, state.table[meld_index])) {
        throw std::invalid_argument("that card does not extend this meld");
    }
    if (state.required_meld_card != -1 && card != state.required_meld_card) {
        throw std::invalid_argument("you must first play the card you took from the pile");
    }
    const int player = state.current_player;
    if (hand_count(player) < 2) {
        throw std::invalid_argument("you must keep a card to discard");
    }

    lay_off_card(player, card, meld_index);
    if (card == state.required_meld_card) {
        state.required_meld_card = -1;
        state.cached_required_meld.clear();
    }
    mask_valid = false;
    return py::make_tuple(0.0f, false);
}

void RummyEnv::opponent_hand(bool* out) const {
    const int opponent = (state.current_player == 1) ? 2 : 1;
    for (int c = 0; c < DECK_SIZE; c++) out[c] = state.card_locations[c] == opponent;
}

py::array_t<bool> RummyEnv::get_opponent_hand() const {
    py::array_t<bool> out(DECK_SIZE);
    opponent_hand(out.mutable_data());
    return out;
}

float RummyEnv::get_score(int player) const {
    return (player == 1) ? state.p1_score : state.p2_score;
}

py::array_t<uint8_t> RummyEnv::get_legal_actions() const {
    const std::array<uint8_t, ACTION_SPACE_SIZE>& mask = legal_mask();
    return py::array_t<uint8_t>(ACTION_SPACE_SIZE, mask.data());
}

py::array_t<float> RummyEnv::get_state() const {
    py::array_t<float> out(OBS_SPACE_SIZE);
    observe(out.mutable_data());
    return out;
}

// --- Thread Pool ---

ThreadPool::ThreadPool(int num_workers) {
    for (int k = 0; k < num_workers; k++) {
        workers.emplace_back([this, k]() { worker_loop(k); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    start_cv.notify_all();
    for (auto& th : workers) th.join();
}

void ThreadPool::worker_loop(int index) {
    uint64_t seen = 0;
    while (true) {
        std::function<void(int, int)> fn;
        {
            std::unique_lock<std::mutex> lock(mutex);
            start_cv.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) return;
            seen = generation;
            fn = task;
        }
        try {
            fn(index, static_cast<int>(workers.size()));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!error) error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (--pending == 0) done_cv.notify_one();
        }
    }
}

void ThreadPool::run(const std::function<void(int, int)>& fn) {
    std::unique_lock<std::mutex> lock(mutex);
    task = fn;
    error = nullptr;
    pending = static_cast<int>(workers.size());
    generation++;
    start_cv.notify_all();
    done_cv.wait(lock, [&] { return pending == 0; });
    if (error) std::rethrow_exception(error);
}

static std::unique_ptr<ThreadPool> make_pool(int num_threads, int num_envs) {
    const int t = std::min(num_threads, num_envs);
    if (t <= 1) return nullptr;
    return std::make_unique<ThreadPool>(t);
}

// --- Env Array ---

void EnvArray::destroy() {
    if (!items) return;
    parallel_for(pool, static_cast<int>(count), [&](int i) { items[i].~RummyEnv(); });
    std::allocator<RummyEnv>().deallocate(items, count);
    items = nullptr;
    count = 0;
}

EnvArray& EnvArray::operator=(EnvArray&& other) noexcept {
    if (this != &other) {
        destroy();
        items = other.items;
        count = other.count;
        pool = other.pool;
        other.items = nullptr;
        other.count = 0;
    }
    return *this;
}

// --- Output buffers ---

// Validates a caller-provided output array (numpy, dtype T, C-contiguous,
// writeable, the given shape; a negative leading dim means "at least
// min_rows" rows) and returns its buffer.
template <typename T>
static T* output_buffer(const py::object& obj, const char* name, std::initializer_list<py::ssize_t> shape,
                        py::ssize_t min_rows = 0) {
    const std::string prefix = std::string(name) + " must be a ";
    if (!py::isinstance<py::array>(obj)) throw std::invalid_argument(prefix + "numpy array");
    py::array a = py::reinterpret_borrow<py::array>(obj);
    // array_t's check uses PyArray_EquivTypes: numpy's own notion of "same
    // dtype", which unifies aliases ('q' and 'l' for int64 on Linux) and
    // rejects a non-native byte order that a type-number test would let through.
    if (!py::isinstance<py::array_t<T>>(a)) {
        throw std::invalid_argument(prefix + py::cast<std::string>(py::str(py::dtype::of<T>())) + " array");
    }
    if (a.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw std::invalid_argument(prefix + std::to_string(shape.size()) + "-D array");
    }
    int d = 0;
    for (py::ssize_t want_dim : shape) {
        const py::ssize_t have = a.shape(d);
        if (want_dim < 0 ? have < min_rows : have != want_dim) {
            throw std::invalid_argument(std::string(name) + ": dimension " + std::to_string(d) + " has size " +
                                        std::to_string(have) + ", need " +
                                        (want_dim < 0 ? "at least " + std::to_string(min_rows)
                                                      : std::to_string(want_dim)));
        }
        d++;
    }
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument(prefix + "C-contiguous array");
    if (!a.writeable()) throw std::invalid_argument(prefix + "writeable array");
    return static_cast<T*>(a.mutable_data());
}

void write_observation(const RummyEnv& env, int i, float* states, bool* masks) {
    env.observe(states + static_cast<size_t>(i) * OBS_SPACE_SIZE);
    const std::array<uint8_t, ACTION_SPACE_SIZE>& mask = env.legal_mask();
    bool* row = masks + static_cast<size_t>(i) * ACTION_SPACE_SIZE;
    for (int a = 0; a < ACTION_SPACE_SIZE; a++) row[a] = mask[a] != 0;
}

// --- Vectorized Environment ---

VectorizedRummyEnv::VectorizedRummyEnv(int num_envs, uint32_t seed, int num_threads, int target_score,
                                       int hand_size) {
    if (num_envs <= 0) throw std::invalid_argument("num_envs must be positive");
    envs.reserve(num_envs);
    for (int i = 0; i < num_envs; i++) {
        envs.emplace_back(seed + static_cast<uint32_t>(i), target_score, DEFAULT_MAX_ROUNDS, hand_size);
    }
    pool = make_pool(num_threads, num_envs);
}

void VectorizedRummyEnv::aux_targets_into(py::object out) {
    const int n = size();
    float* o = output_buffer<float>(out, "out", {n, AUX_SPACE_SIZE});
    py::gil_scoped_release release;
    parallel_for(pool.get(), n, [&](int i) {
        envs[i].aux_targets(o + static_cast<size_t>(i) * AUX_SPACE_SIZE);
    });
}

py::array_t<float> VectorizedRummyEnv::aux_targets() {
    py::array_t<float> out({size(), AUX_SPACE_SIZE});
    aux_targets_into(out);
    return out;
}

py::array_t<int32_t> VectorizedRummyEnv::current_players() const {
    py::array_t<int32_t> out(size());
    int32_t* o = out.mutable_data();
    for (int i = 0; i < size(); i++) o[i] = envs[i].get_current_player();
    return out;
}

py::array_t<bool> VectorizedRummyEnv::opponent_hands() const {
    py::array_t<bool> out({size(), DECK_SIZE});
    bool* o = out.mutable_data();
    for (int i = 0; i < size(); i++) envs[i].opponent_hand(o + static_cast<size_t>(i) * DECK_SIZE);
    return out;
}

py::tuple VectorizedRummyEnv::reset() {
    const int n = size();
    py::array_t<float> states({n, OBS_SPACE_SIZE});
    py::array_t<bool> masks({n, ACTION_SPACE_SIZE});
    float* s = states.mutable_data();
    bool* m = masks.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), n, [&](int i) {
            envs[i].reset();
            write_observation(envs[i], i, s, m);
        });
    }
    return py::make_tuple(states, masks);
}

void VectorizedRummyEnv::step_raw(const int64_t* a, float* s, bool* m, float* r, bool* d, bool* e, int32_t* w) {
    py::gil_scoped_release release;
    parallel_for(pool.get(), size(), [&](int i) {
        auto result = envs[i].step_raw(static_cast<int>(a[i]));
        e[i] = envs[i].round_ended();
        w[i] = e[i] ? envs[i].get_last_round_out() : 0;
        if (result.second) envs[i].reset();
        r[i] = result.first;
        d[i] = result.second;
        write_observation(envs[i], i, s, m);
    });
}

py::tuple VectorizedRummyEnv::step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
    const int n = size();
    py::array_t<float> states({n, OBS_SPACE_SIZE});
    py::array_t<bool> masks({n, ACTION_SPACE_SIZE});
    py::array_t<float> rewards(n);
    py::array_t<bool> dones(n);
    py::array_t<bool> round_ends(n);
    py::array_t<int32_t> round_outs(n);
    step_into(actions, states, masks, rewards, dones, round_ends, round_outs);
    return py::make_tuple(states, masks, rewards, dones, round_ends, round_outs);
}

void VectorizedRummyEnv::step_into(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions,
                                   py::object states, py::object masks, py::object rewards, py::object dones,
                                   py::object round_ends, py::object round_outs) {
    const int n = size();
    if (actions.ndim() != 1 || actions.shape(0) != n) {
        throw std::invalid_argument("actions must be a 1-D array with one entry per environment");
    }
    float* s = output_buffer<float>(states, "states", {n, OBS_SPACE_SIZE});
    bool* m = output_buffer<bool>(masks, "masks", {n, ACTION_SPACE_SIZE});
    float* r = output_buffer<float>(rewards, "rewards", {n});
    bool* d = output_buffer<bool>(dones, "dones", {n});
    bool* e = output_buffer<bool>(round_ends, "round_ends", {n});
    int32_t* w = output_buffer<int32_t>(round_outs, "round_outs", {n});
    step_raw(actions.data(), s, m, r, d, e, w);
}

// --- Env Batch ---

void EnvBatch::init_alive() {
    if (envs.size() == 0) throw std::invalid_argument("EnvBatch needs at least one environment");
    alive.assign(envs.size(), 1);
    for (size_t i = 0; i < envs.size(); i++) alive[i] = envs[i].is_done() ? 0 : 1;
}

EnvBatch::EnvBatch(EnvArray&& sources, std::unique_ptr<ThreadPool>&& pool_)
    : pool(std::move(pool_)), envs(std::move(sources)) {
    init_alive();
}

// The pool member is declared first so the engines can be copied onto it here.
EnvBatch::EnvBatch(const std::vector<RummyEnv>& sources, int num_threads)
    : pool(make_pool(num_threads, static_cast<int>(sources.size()))),
      envs(sources.size(), pool.get(), static_cast<int>(sources.size()),
           [&](int i, auto& emplace) { emplace(i, sources[i]); }) {
    init_alive();
}

EnvBatch::EnvBatch(std::vector<RummyEnv>&& sources, int num_threads)
    : pool(make_pool(num_threads, static_cast<int>(sources.size()))),
      envs(sources.size(), pool.get(), static_cast<int>(sources.size()),
           [&](int i, auto& emplace) { emplace(i, std::move(sources[i])); }) {
    init_alive();
}

std::unique_ptr<EnvBatch> EnvBatch::for_search(
        const std::vector<RummyEnv>& sources,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> repeats,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds,
        py::object weights, int num_threads) {
    const int n = static_cast<int>(sources.size());
    if (repeats.ndim() != 1 || repeats.shape(0) != n) {
        throw std::invalid_argument("repeats must have one entry per source game");
    }
    if (seeds.ndim() != 2 || seeds.shape(0) != n) {
        throw std::invalid_argument("seeds must be [num_sources, worlds]");
    }
    const int worlds = static_cast<int>(seeds.shape(1));
    const float* w = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> weights_arr;
    if (!weights.is_none()) {
        weights_arr = weights.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        if (weights_arr.ndim() != 2 || weights_arr.shape(0) != n || weights_arr.shape(1) != DECK_SIZE) {
            throw std::invalid_argument("weights must be [num_sources, 52]");
        }
        w = weights_arr.data();
    }
    const int32_t* rep = repeats.data();
    const uint32_t* sd = seeds.data();

    // Sim j*rep[i] .. of world j = (source i, redeal k) are contiguous.
    const int nw = n * worlds;
    std::vector<size_t> first(nw + 1, 0);
    for (int j = 0; j < nw; j++) first[j + 1] = first[j] + static_cast<size_t>(std::max(0, rep[j / worlds]));
    const size_t total = first[nw];
    std::unique_ptr<ThreadPool> pool = make_pool(num_threads, static_cast<int>(total));

    EnvArray sims;
    {
        py::gil_scoped_release release;
        EnvArray world_copies(nw, pool.get(), nw, [&](int j, auto& emplace) {
            const int i = j / worlds;
            RummyEnv& world = emplace(j, sources[i]);
            world.set_manual_meld(0);   // simulated players all auto-meld, as in training
            if (w) world.randomize_hidden_weighted(sd[j], w + i * DECK_SIZE);
            else world.randomize_hidden(sd[j]);
        });
        // Each world's group copies it per candidate; the last copy is a move,
        // which is safe because every copy of that world was made by this group.
        sims = EnvArray(total, pool.get(), nw, [&](int j, auto& emplace) {
            for (size_t s = first[j]; s + 1 < first[j + 1]; s++) emplace(s, world_copies[j]);
            if (first[j + 1] > first[j]) emplace(first[j + 1] - 1, std::move(world_copies[j]));
        });
    }
    return std::unique_ptr<EnvBatch>(new EnvBatch(std::move(sims), std::move(pool)));
}

RummyEnv EnvBatch::get(int i) const {
    if (i < 0 || i >= size()) throw std::out_of_range("game index out of range");
    return envs[i];
}

std::vector<int> EnvBatch::alive_indices() const {
    std::vector<int> idx;
    idx.reserve(envs.size());
    for (int i = 0; i < size(); i++) if (alive[i]) idx.push_back(i);
    return idx;
}

void EnvBatch::observe_alive_raw(const std::vector<int>& idx, float* s, bool* m, int32_t* p, int64_t* ix,
                                 int8_t* ph) {
    py::gil_scoped_release release;
    parallel_for(pool.get(), static_cast<int>(idx.size()), [&](int j) {
        const RummyEnv& env = envs[idx[j]];
        write_observation(env, j, s, m);
        p[j] = env.get_current_player();
        ix[j] = idx[j];
        if (ph) ph[j] = env.in_discard_phase() ? 1 : 0;
    });
}

py::tuple EnvBatch::observe_alive() {
    const std::vector<int> idx = alive_indices();
    const int k = static_cast<int>(idx.size());
    py::array_t<float> states({k, OBS_SPACE_SIZE});
    py::array_t<bool> masks({k, ACTION_SPACE_SIZE});
    py::array_t<int32_t> players(k);
    py::array_t<int64_t> indices(k);
    observe_alive_raw(idx, states.mutable_data(), masks.mutable_data(), players.mutable_data(),
                      indices.mutable_data(), nullptr);
    return py::make_tuple(states, masks, players, indices);
}

int EnvBatch::observe_alive_into(py::object states, py::object masks, py::object players, py::object indices,
                                 py::object phases) {
    const std::vector<int> idx = alive_indices();
    const int k = static_cast<int>(idx.size());
    float* s = output_buffer<float>(states, "states", {-1, OBS_SPACE_SIZE}, k);
    bool* m = output_buffer<bool>(masks, "masks", {-1, ACTION_SPACE_SIZE}, k);
    int32_t* p = output_buffer<int32_t>(players, "players", {-1}, k);
    int64_t* ix = output_buffer<int64_t>(indices, "indices", {-1}, k);
    int8_t* ph = output_buffer<int8_t>(phases, "phases", {-1}, k);
    observe_alive_raw(idx, s, m, p, ix, ph);
    return k;
}

py::array_t<bool> EnvBatch::alive_mask() const {
    py::array_t<bool> out(size());
    bool* o = out.mutable_data();
    for (int i = 0; i < size(); i++) o[i] = alive[i] != 0;
    return out;
}

py::tuple EnvBatch::observe() {
    const int n = size();
    py::array_t<float> states({n, OBS_SPACE_SIZE});
    py::array_t<bool> masks({n, ACTION_SPACE_SIZE});
    py::array_t<int32_t> players(n);
    float* s = states.mutable_data();
    bool* m = masks.mutable_data();
    int32_t* p = players.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), n, [&](int i) {
            write_observation(envs[i], i, s, m);
            p[i] = envs[i].get_current_player();
        });
    }
    return py::make_tuple(states, masks, players);
}

py::tuple EnvBatch::step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
    const int n = size();
    if (actions.ndim() != 1 || actions.shape(0) != n) {
        throw std::invalid_argument("actions must be a 1-D array with one entry per environment");
    }
    const int64_t* a = actions.data();
    py::array_t<float> rewards(n);
    py::array_t<bool> dones(n);
    py::array_t<bool> round_ends(n);
    float* r = rewards.mutable_data();
    bool* d = dones.mutable_data();
    bool* e = round_ends.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), n, [&](int i) {
            if (!alive[i]) {
                r[i] = 0.0f;
                d[i] = true;
                e[i] = false;
                return;
            }
            auto result = envs[i].step_raw(static_cast<int>(a[i]));
            r[i] = result.first;
            d[i] = result.second;
            e[i] = envs[i].round_ended();
            if (result.second) alive[i] = 0;
        });
    }
    return py::make_tuple(rewards, dones, round_ends);
}

void EnvBatch::halt(py::array_t<bool, py::array::c_style | py::array::forcecast> which) {
    if (which.ndim() != 1 || which.shape(0) != size()) {
        throw std::invalid_argument("which must be a 1-D bool array with one entry per environment");
    }
    const bool* w = which.data();
    for (int i = 0; i < size(); i++) if (w[i]) alive[i] = 0;
}

std::unique_ptr<EnvBatch> EnvBatch::expand(py::array_t<int32_t, py::array::c_style | py::array::forcecast> repeats) const {
    const int n = size();
    if (repeats.ndim() != 1 || repeats.shape(0) != n) {
        throw std::invalid_argument("repeats must have one entry per environment");
    }
    const int32_t* rep = repeats.data();
    std::vector<size_t> first(n + 1, 0);
    for (int i = 0; i < n; i++) first[i + 1] = first[i] + static_cast<size_t>(std::max(0, rep[i]));
    const size_t total = first[n];
    std::vector<uint8_t> alive_copy(total);
    for (int i = 0; i < n; i++) std::fill(alive_copy.begin() + first[i], alive_copy.begin() + first[i + 1], alive[i]);

    std::unique_ptr<ThreadPool> new_pool = make_pool(pool ? pool->size() : 1, static_cast<int>(total));
    EnvArray copies;
    {
        py::gil_scoped_release release;
        copies = EnvArray(total, new_pool.get(), n, [&](int i, auto& emplace) {
            for (size_t s = first[i]; s < first[i + 1]; s++) emplace(s, envs[i]);
        });
    }
    std::unique_ptr<EnvBatch> out(new EnvBatch(std::move(copies), std::move(new_pool)));
    out->alive = std::move(alive_copy);
    return out;
}

py::array_t<int32_t> EnvBatch::penalised() const {
    py::array_t<int32_t> out(size());
    int32_t* o = out.mutable_data();
    for (int i = 0; i < size(); i++) o[i] = envs[i].get_penalised();
    return out;
}

py::array_t<float> EnvBatch::scores() const {
    py::array_t<float> out({size(), 2});
    float* o = out.mutable_data();
    for (int i = 0; i < size(); i++) {
        o[2 * i] = envs[i].get_score(1);
        o[2 * i + 1] = envs[i].get_score(2);
    }
    return out;
}

void EnvBatch::randomize_hidden(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds) {
    const int n = size();
    if (seeds.ndim() != 1 || seeds.shape(0) != n) {
        throw std::invalid_argument("seeds must be a 1-D array with one entry per environment");
    }
    const uint32_t* sd = seeds.data();
    py::gil_scoped_release release;
    parallel_for(pool.get(), n, [&](int i) { envs[i].randomize_hidden(sd[i]); });
}

void EnvBatch::randomize_hidden_weighted(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds,
                                         py::array_t<float, py::array::c_style | py::array::forcecast> weights) {
    const int n = size();
    if (seeds.ndim() != 1 || seeds.shape(0) != n) {
        throw std::invalid_argument("seeds must be a 1-D array with one entry per environment");
    }
    if (weights.ndim() != 2 || weights.shape(0) != n || weights.shape(1) != DECK_SIZE) {
        throw std::invalid_argument("weights must be [N, 52]");
    }
    const uint32_t* sd = seeds.data();
    const float* w = weights.data();
    py::gil_scoped_release release;
    parallel_for(pool.get(), n, [&](int i) {
        envs[i].randomize_hidden_weighted(sd[i], w + static_cast<size_t>(i) * DECK_SIZE);
    });
}

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(rummy_engine, m) {
    m.attr("OBS_SPACE_SIZE") = OBS_SPACE_SIZE;
    m.attr("AUX_SPACE_SIZE") = AUX_SPACE_SIZE;
    m.attr("HISTORY_LEN") = HISTORY_LEN;
    m.attr("EVENT_DIM") = EVENT_DIM;

    py::class_<RummyEnv>(m, "RummyEnv")
        .def(py::init<uint32_t, int, int, int>(), py::arg("seed"), py::arg("target_score") = DEFAULT_TARGET_SCORE,
             py::arg("max_rounds") = DEFAULT_MAX_ROUNDS, py::arg("hand_size") = DEFAULT_HAND_SIZE)
        .def("reset", &RummyEnv::reset)
        .def("step", &RummyEnv::step,
             "Returns (reward, done). done is true only when the game ends (a player reached the target score).")
        .def("is_done", &RummyEnv::is_done)
        .def("get_score", &RummyEnv::get_score)
        .def("get_round", &RummyEnv::get_round, "Current round number, from 1.")
        .def("round_ended", &RummyEnv::round_ended, "True if the last step ended a round.")
        .def("get_target_score", &RummyEnv::get_target_score)
        .def("get_penalised", &RummyEnv::get_penalised,
             "Seat that broke the pile-draw obligation and lost the game (0 = none).")
        .def("get_required_card", &RummyEnv::get_required_card,
             "Card taken from the pile that must be melded or laid off this turn, or -1.")
        .def("get_legal_actions", &RummyEnv::get_legal_actions)
        .def("get_state", &RummyEnv::get_state)
        .def("get_current_player", &RummyEnv::get_current_player)
        .def("get_opponent_hand", &RummyEnv::get_opponent_hand)
        .def("get_aux_targets", &RummyEnv::get_aux_targets,
             "Ground-truth auxiliary targets for the player to move (training only).")
        .def("get_history", &RummyEnv::get_history, "This round's (player, kind, card, count) events, oldest first.")
        .def("get_last_round_out", &RummyEnv::get_last_round_out)
        .def("get_hand_size", &RummyEnv::get_hand_size)
        .def("randomize_hidden", &RummyEnv::randomize_hidden, py::arg("seed"))
        .def("randomize_hidden_weighted", &RummyEnv::randomize_hidden_weighted_py, py::arg("seed"), py::arg("weights"))
        .def("set_manual_meld", &RummyEnv::set_manual_meld, py::arg("player"),
             "Seat that lays down its own melds (0 = everyone auto-melds).")
        .def("get_manual_meld", &RummyEnv::get_manual_meld)
        .def("is_valid_meld", &RummyEnv::is_valid_meld, py::arg("cards"))
        .def("meld", &RummyEnv::meld, py::arg("cards"), "Lay down a set or run from hand; returns (reward, done).")
        .def("lay_off", &RummyEnv::lay_off, py::arg("card"), py::arg("meld_index"),
             "Add a hand card to table meld meld_index; returns (reward, done).")
        .def("can_lay_off", &RummyEnv::can_lay_off, py::arg("card"), py::arg("meld"))
        .def("get_table", &RummyEnv::get_table, "Melds on the table as lists of cards, oldest first.")
        .def("clone", [](const RummyEnv& env) { return RummyEnv(env); });

    py::class_<VectorizedRummyEnv>(m, "VectorizedRummyEnv")
        .def(py::init<int, uint32_t, int, int, int>(), py::arg("num_envs"), py::arg("seed"),
             py::arg("num_threads") = 1, py::arg("target_score") = DEFAULT_TARGET_SCORE,
             py::arg("hand_size") = DEFAULT_HAND_SIZE)
        .def("aux_targets", &VectorizedRummyEnv::aux_targets,
             "[N,AUX] float32 auxiliary targets for the player to move (training only).")
        .def("aux_targets_into", &VectorizedRummyEnv::aux_targets_into, py::arg("out"),
             "aux_targets() written into a float32 [N,AUX] C-contiguous array.")
        .def_property_readonly("num_envs", &VectorizedRummyEnv::size)
        .def("get", &VectorizedRummyEnv::get, "Copy of live game i.")
        .def("current_players", &VectorizedRummyEnv::current_players, "Player to move in each game (1 or 2).")
        .def("opponent_hands", &VectorizedRummyEnv::opponent_hands,
             "[N,52] bool: cards held by the player not to move (training target, not observable).")
        .def("reset", &VectorizedRummyEnv::reset, "Returns (states[N,obs], masks[N,105] bool).")
        .def("step", &VectorizedRummyEnv::step,
             "Returns (states, masks, rewards[N] float32, dones[N] bool, round_ends[N] bool, "
             "round_outs[N] int32: player who went out, 0 otherwise); finished games are auto-reset.")
        .def("step_into", &VectorizedRummyEnv::step_into, py::arg("actions"), py::arg("states"), py::arg("masks"),
             py::arg("rewards"), py::arg("dones"), py::arg("round_ends"), py::arg("round_outs"),
             "step() written into caller-provided C-contiguous arrays of exact shape and dtype: states float32 "
             "[N,obs], masks bool [N,105], rewards float32 [N], dones bool [N], round_ends bool [N], "
             "round_outs int32 [N].");

    py::class_<EnvBatch>(m, "EnvBatch")
        .def(py::init<const std::vector<RummyEnv>&, int>(), py::arg("envs"), py::arg("num_threads") = 1)
        .def_property_readonly("size", &EnvBatch::size)
        .def("get", &EnvBatch::get, "Copy of game i.")
        .def("alive", &EnvBatch::alive_mask)
        .def("observe", &EnvBatch::observe, "Returns (states, masks, current_players[N] int32).")
        .def("observe_alive", &EnvBatch::observe_alive,
             "Live games only: (states[K,obs], masks[K,105], current_players[K], indices[K]).")
        .def("observe_alive_into", &EnvBatch::observe_alive_into, py::arg("states"), py::arg("masks"),
             py::arg("players"), py::arg("indices"), py::arg("phases"),
             "observe_alive() written into the first K rows of caller-provided C-contiguous arrays with at "
             "least K rows: states float32 [cap,obs], masks bool [cap,105], players int32 [cap], indices int64 "
             "[cap], phases int8 [cap] (1 in the discard phase, i.e. obs[-3]). Rows K.. are left untouched; "
             "returns K.")
        .def_static("for_search", &EnvBatch::for_search, py::arg("sources"), py::arg("repeats"),
                    py::arg("seeds"), py::arg("weights") = py::none(), py::arg("num_threads") = 1,
                    "Rollout batch for determinized search: per source, worlds x repeats copies with hidden "
                    "cards redealt (order: source, world, repeat).")
        .def("step", &EnvBatch::step,
             "Returns (rewards, dones, round_ends); finished games are left as they are.")
        .def("scores", &EnvBatch::scores, "Returns [N,2] scores for players 1 and 2.")
        .def("penalised", &EnvBatch::penalised, "Per game: seat that broke the pile-draw obligation, or 0.")
        .def("halt", &EnvBatch::halt, py::arg("which"), "Stop stepping the flagged games.")
        .def("expand", &EnvBatch::expand, py::arg("repeats"),
             "New batch with game i repeated repeats[i] times, halted state included.")
        .def("randomize_hidden", &EnvBatch::randomize_hidden, py::arg("seeds"))
        .def("randomize_hidden_weighted", &EnvBatch::randomize_hidden_weighted, py::arg("seeds"), py::arg("weights"));
}
