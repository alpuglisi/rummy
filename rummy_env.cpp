#include "rummy_env.h"

// --- Helper Functions ---

int RummyEnv::get_point_value(int card) const {
    int rank = get_rank(card);
    if (rank == 0) return 1;   // Ace
    if (rank >= 9) return 10;  // 10, J, Q, K
    return rank + 1;           // 2-9 at face value
}

std::vector<Meld> RummyEnv::find_all_possible_melds(const std::vector<int>& hand) const {
    std::vector<Meld> valid_melds;

    // 1. Sets (3 or 4 of a kind)
    for (int rank = 0; rank < 13; rank++) {
        Meld current_set;
        for (int card : hand) {
            if (get_rank(card) == rank) current_set.push_back(card);
        }
        if (current_set.size() >= 3) valid_melds.push_back(current_set);
    }

    // 2. Runs (3+ consecutive in suit)
    for (int suit = 0; suit < 4; suit++) {
        std::vector<int> suit_cards;
        for (int card : hand) {
            if (get_suit(card) == suit) suit_cards.push_back(card);
        }
        std::sort(suit_cards.begin(), suit_cards.end());

        for (size_t i = 0; i < suit_cards.size(); i++) {
            Meld current_run = { suit_cards[i] };
            for (size_t j = i + 1; j < suit_cards.size(); j++) {
                if (get_rank(suit_cards[j]) == get_rank(current_run.back()) + 1) {
                    current_run.push_back(suit_cards[j]);
                    if (current_run.size() >= 3) valid_melds.push_back(current_run);
                } else {
                    break;
                }
            }
        }
    }
    return valid_melds;
}

Meld RummyEnv::find_largest_meld_with_card(const std::vector<int>& hand, int card) const {
    std::vector<Meld> melds = find_all_possible_melds(hand);
    Meld best;
    for (const Meld& m : melds) {
        if (std::find(m.begin(), m.end(), card) != m.end() && m.size() > best.size()) {
            best = m;
        }
    }
    return best;
}

bool RummyEnv::can_take_pile(int player, size_t index) const {
    const int card = state.discard_pile[index];
    std::vector<int> combined = get_hand(player);
    combined.insert(combined.end(), state.discard_pile.begin() + index, state.discard_pile.end());
    Meld m = find_largest_meld_with_card(combined, card);
    // A fresh meld needs a 4th card left in hand (a larger meld is trimmed to
    // leave a discard, see the discard phase); otherwise the card may extend
    // a meld already on the table.
    return (!m.empty() && combined.size() >= 4) || (find_lay_off(card) >= 0 && combined.size() > 1);
}

int RummyEnv::immediate_points(const std::vector<int>& hand, int card) const {
    std::vector<int> combined(hand);
    combined.push_back(card);
    Meld m = find_largest_meld_with_card(combined, card);
    if (!m.empty() && combined.size() >= 4) {
        int pts = 0;
        for (int c : m) pts += get_point_value(c);
        return pts;
    }
    if (find_lay_off(card) >= 0 && combined.size() > 1) return get_point_value(card);
    return 0;
}

bool RummyEnv::could_go_out(int player, const std::vector<int>& extra) const {
    // Simulate in place and restore: auto_meld only touches card locations,
    // the table and the scores, and copying the whole engine (observation
    // buffer included) per simulated draw was the cost of aux_targets().
    RummyEnv& self = const_cast<RummyEnv&>(*this);
    const auto saved_locations = state.card_locations;
    const auto saved_table = state.table;
    const float saved_p1 = state.p1_score, saved_p2 = state.p2_score;
    for (int c : extra) self.state.card_locations[c] = static_cast<int8_t>(player);
    self.auto_meld(player);
    const bool out = get_hand(player).size() <= 1;
    self.state.card_locations = saved_locations;
    self.state.table = saved_table;
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
    const std::vector<int> my_hand = get_hand(me);
    const std::vector<int> opp_hand = get_hand(opp);
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
    scalars[0] = find_all_possible_melds(opp_hand).empty() ? 0.0f : 1.0f;
    int opp_points = 0;
    for (int c : opp_hand) opp_points += get_point_value(c);
    scalars[1] = static_cast<float>(opp_points) / 100.0f;

    // Can the opponent go out on their next turn? Try the deck's next card
    // and every pile take that would be legal for them.
    bool go_out = false;
    if (deck_index < DECK_SIZE) go_out = could_go_out(opp, {deck_order[deck_index]});
    for (size_t i = 0; i < state.discard_pile.size() && !go_out; i++) {
        if (can_take_pile(opp, i)) {
            std::vector<int> taken(state.discard_pile.begin() + i, state.discard_pile.end());
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

std::vector<int> RummyEnv::get_hand(int player) const {
    std::vector<int> hand;
    for (int i = 0; i < DECK_SIZE; i++) {
        if (state.card_locations[i] == player) hand.push_back(i);
    }
    return hand;
}

void RummyEnv::auto_meld(int player) {
    bool found_meld = true;
    while (found_meld) {
        found_meld = false;
        std::vector<int> hand = get_hand(player);
        std::vector<Meld> possible_melds = find_all_possible_melds(hand);

        if (!possible_melds.empty()) {
            // Greedily pick the highest scoring meld
            Meld best_meld = possible_melds[0];
            int best_score = 0;

            for (const Meld& m : possible_melds) {
                int current_score = 0;
                for (int c : m) current_score += get_point_value(c);
                if (current_score > best_score) {
                    best_score = current_score;
                    best_meld = m;
                }
            }

            place_meld(player, best_meld);
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

void RummyEnv::place_meld(int player, const Meld& m) {
    Meld sorted(m);
    std::sort(sorted.begin(), sorted.end());
    for (int c : sorted) {
        state.card_locations[c] = 4;
        if (player == 1) state.p1_score += get_point_value(c);
        else state.p2_score += get_point_value(c);
    }
    state.table.push_back(sorted);
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
    deck_order.resize(DECK_SIZE);
    for (int i = 0; i < DECK_SIZE; i++) deck_order[i] = i;
    observation_buffer.resize(OBS_SPACE_SIZE, 0.0f);
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
    update_observation_buffer();
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
}

float RummyEnv::end_round(int acting_player) {
    state.last_round_out = get_hand(acting_player).empty() ? acting_player : 0;
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
    return reward;
}

void RummyEnv::update_observation_buffer() {
    std::fill(observation_buffer.begin(), observation_buffer.end(), 0.0f);

    for (int i = 0; i < DECK_SIZE; i++) {
        // Channel 1: Hand
        if (state.card_locations[i] == state.current_player) {
            observation_buffer[i] = 1.0f;
        }
    }

    // Channel 2 & 3: Discard Presence and Depth/Order
    for (size_t i = 0; i < state.discard_pile.size(); i++) {
        int card = state.discard_pile[i];
        observation_buffer[DECK_SIZE + card] = 1.0f;
        observation_buffer[DECK_SIZE * 2 + card] = static_cast<float>(i + 1) / state.discard_pile.size();
    }

    // Channel 4, 5 & 6: melded cards (dead for everyone), opponent's known hand
    // cards, and unseen cards (deck or opponent's unknown hand: anything the
    // current player has no information about and could still be drawn).
    const int opponent = (state.current_player == 1) ? 2 : 1;
    int own_hand_size = 0, opp_hand_size = 0;
    for (int i = 0; i < DECK_SIZE; i++) {
        if (state.card_locations[i] == 4) observation_buffer[DECK_SIZE * 3 + i] = 1.0f;
        if (state.card_locations[i] == opponent) {
            opp_hand_size++;
            if (state.publicly_known[i]) observation_buffer[DECK_SIZE * 4 + i] = 1.0f;
            else observation_buffer[DECK_SIZE * 5 + i] = 1.0f;
        }
        if (state.card_locations[i] == 0) observation_buffer[DECK_SIZE * 5 + i] = 1.0f;
        if (state.card_locations[i] == state.current_player) own_hand_size++;
    }

    // Turn history: the last HISTORY_LEN events, most recent in the last slot.
    const int base = DECK_SIZE * 6;
    const int n_events = static_cast<int>(state.history.size());
    const int first = std::max(0, n_events - HISTORY_LEN);
    for (int e = first; e < n_events; e++) {
        const Event& ev = state.history[e];
        float* slot = &observation_buffer[base + (HISTORY_LEN - (n_events - e)) * EVENT_DIM];
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
    observation_buffer[OBS_SPACE_SIZE - 8] = own_score / static_cast<float>(target_score);
    observation_buffer[OBS_SPACE_SIZE - 7] = opp_score / static_cast<float>(target_score);
    observation_buffer[OBS_SPACE_SIZE - 6] = static_cast<float>(own_hand_size) / 26.0f;
    observation_buffer[OBS_SPACE_SIZE - 5] = static_cast<float>(opp_hand_size) / 26.0f;
    observation_buffer[OBS_SPACE_SIZE - 4] = (own_score - opp_score) / 100.0f;

    // Meta variables
    observation_buffer[OBS_SPACE_SIZE - 3] = state.turn_phase_is_discard ? 1.0f : 0.0f;
    observation_buffer[OBS_SPACE_SIZE - 2] = static_cast<float>(deck_index) / DECK_SIZE;
    observation_buffer[OBS_SPACE_SIZE - 1] = (state.required_meld_card != -1) ? 1.0f : 0.0f;
}

std::vector<uint8_t> RummyEnv::compute_legal_mask() {
    std::vector<uint8_t> mask(ACTION_SPACE_SIZE, 0);
    if (state.is_terminal) return mask;

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
        std::vector<int> consumed_cards;
        if (state.required_meld_card != -1) {
            std::vector<int> hand = get_hand(state.current_player);
            Meld m = find_largest_meld_with_card(hand, state.required_meld_card);
            if (!m.empty() && m.size() == hand.size()) {
                // Melding everything would leave nothing to discard: drop one
                // end card of the meld (never the required card) so the
                // player keeps a discard, or fall back to a lay-off when the
                // meld is only three cards.
                if (m.size() > 3) {
                    std::sort(m.begin(), m.end());
                    if (m.front() != state.required_meld_card) m.erase(m.begin());
                    else m.pop_back();
                } else {
                    m.clear();
                }
            }
            state.cached_required_meld = m;
            if (m.empty() && find_lay_off(state.required_meld_card) >= 0) {
                // The pile card extends a table meld instead: it is laid off
                // at discard time and cannot itself be discarded.
                consumed_cards.push_back(state.required_meld_card);
            } else {
                consumed_cards = m;
            }
        }

        bool has_legal_discard = false;
        for (int i = 0; i < DECK_SIZE; i++) {
            if (state.card_locations[i] == state.current_player) {
                if (std::find(consumed_cards.begin(), consumed_cards.end(), i) == consumed_cards.end()) {
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
    return mask;
}

bool RummyEnv::is_legal_action(int action) {
    if (action < 0 || action >= ACTION_SPACE_SIZE) return false;
    std::vector<uint8_t> mask = compute_legal_mask();
    return mask[action] == 1;
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

    place_meld(player, state.cached_required_meld);
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
        update_observation_buffer();
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
                update_observation_buffer();
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
        if (get_hand(acting_player).empty() || deck_index >= DECK_SIZE) {
            float reward = end_round(acting_player);
            update_observation_buffer();
            return {reward, state.is_terminal};
        }

        state.turn_phase_is_discard = false;
        state.current_player = (acting_player == 1) ? 2 : 1;

        update_observation_buffer();
        return {0.1f, false};
    }
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
            deck_order[deck_index + (k - unknown_in_hand)] = pool[k];
        }
    }
    rng.seed(seed ^ 0x9e3779b9u);
    update_observation_buffer();
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
    for (size_t k = 0; k < rest.size(); k++) deck_order[deck_index + k] = rest[k];
    rng.seed(seed ^ 0x9e3779b9u);
    update_observation_buffer();
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
    if (cards.size() >= get_hand(player).size()) {
        throw std::invalid_argument("you must keep a card to discard");
    }

    place_meld(player, cards);
    if (std::find(cards.begin(), cards.end(), state.required_meld_card) != cards.end()) {
        state.required_meld_card = -1;
        state.cached_required_meld.clear();
    }
    update_observation_buffer();
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
    if (get_hand(player).size() < 2) {
        throw std::invalid_argument("you must keep a card to discard");
    }

    lay_off_card(player, card, meld_index);
    if (card == state.required_meld_card) {
        state.required_meld_card = -1;
        state.cached_required_meld.clear();
    }
    update_observation_buffer();
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

py::array_t<uint8_t> RummyEnv::get_legal_actions() {
    std::vector<uint8_t> mask = compute_legal_mask();
    return py::array_t<uint8_t>(mask.size(), mask.data());
}

py::array_t<float> RummyEnv::get_state() const {
    return py::array_t<float>(observation_buffer.size(), observation_buffer.data());
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

void write_observation(const RummyEnv& env, int i, float* states, bool* masks) {
    const std::vector<float>& obs = env.observation();
    std::copy(obs.begin(), obs.end(), states + static_cast<size_t>(i) * OBS_SPACE_SIZE);
    std::vector<uint8_t> mask = const_cast<RummyEnv&>(env).legal_mask();
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

py::array_t<float> VectorizedRummyEnv::aux_targets() {
    const int n = size();
    py::array_t<float> out({n, AUX_SPACE_SIZE});
    float* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), n, [&](int i) {
            envs[i].aux_targets(o + static_cast<size_t>(i) * AUX_SPACE_SIZE);
        });
    }
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

py::tuple VectorizedRummyEnv::step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
    const int n = size();
    if (actions.ndim() != 1 || actions.shape(0) != n) {
        throw std::invalid_argument("actions must be a 1-D array with one entry per environment");
    }
    const int64_t* a = actions.data();

    py::array_t<float> states({n, OBS_SPACE_SIZE});
    py::array_t<bool> masks({n, ACTION_SPACE_SIZE});
    py::array_t<float> rewards(n);
    py::array_t<bool> dones(n);
    py::array_t<bool> round_ends(n);
    py::array_t<int32_t> round_outs(n);
    float* s = states.mutable_data();
    bool* m = masks.mutable_data();
    float* r = rewards.mutable_data();
    bool* d = dones.mutable_data();
    bool* e = round_ends.mutable_data();
    int32_t* w = round_outs.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), n, [&](int i) {
            auto result = envs[i].step_raw(static_cast<int>(a[i]));
            e[i] = envs[i].round_ended();
            w[i] = e[i] ? envs[i].get_last_round_out() : 0;
            if (result.second) envs[i].reset();
            r[i] = result.first;
            d[i] = result.second;
            write_observation(envs[i], i, s, m);
        });
    }
    return py::make_tuple(states, masks, rewards, dones, round_ends, round_outs);
}

// --- Env Batch ---

EnvBatch::EnvBatch(const std::vector<RummyEnv>& sources, int num_threads)
    : envs(sources), alive(sources.size(), 1) {
    if (envs.empty()) throw std::invalid_argument("EnvBatch needs at least one environment");
    for (size_t i = 0; i < envs.size(); i++) alive[i] = envs[i].is_done() ? 0 : 1;
    pool = make_pool(num_threads, static_cast<int>(envs.size()));
}

EnvBatch::EnvBatch(std::vector<RummyEnv>&& sources, int num_threads)
    : envs(std::move(sources)), alive(envs.size(), 1) {
    if (envs.empty()) throw std::invalid_argument("EnvBatch needs at least one environment");
    for (size_t i = 0; i < envs.size(); i++) alive[i] = envs[i].is_done() ? 0 : 1;
    pool = make_pool(num_threads, static_cast<int>(envs.size()));
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

    std::vector<RummyEnv> sims;
    size_t total = 0;
    for (int i = 0; i < n; i++) total += static_cast<size_t>(worlds) * std::max(0, rep[i]);
    sims.reserve(total);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < worlds; k++) {
            RummyEnv world = sources[i];
            world.set_manual_meld(0);   // simulated players all auto-meld, as in training
            if (w) world.randomize_hidden_weighted(sd[i * worlds + k], w + i * DECK_SIZE);
            else world.randomize_hidden(sd[i * worlds + k]);
            for (int r = 0; r < rep[i]; r++) sims.push_back(world);
        }
    }
    return std::unique_ptr<EnvBatch>(new EnvBatch(std::move(sims), num_threads));
}

py::tuple EnvBatch::observe_alive() {
    std::vector<int> idx;
    idx.reserve(envs.size());
    for (int i = 0; i < size(); i++) if (alive[i]) idx.push_back(i);
    const int k = static_cast<int>(idx.size());
    py::array_t<float> states({k, OBS_SPACE_SIZE});
    py::array_t<bool> masks({k, ACTION_SPACE_SIZE});
    py::array_t<int32_t> players(k);
    py::array_t<int64_t> indices(k);
    float* s = states.mutable_data();
    bool* m = masks.mutable_data();
    int32_t* p = players.mutable_data();
    int64_t* ix = indices.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(pool.get(), k, [&](int j) {
            write_observation(envs[idx[j]], j, s, m);
            p[j] = envs[idx[j]].get_current_player();
            ix[j] = idx[j];
        });
    }
    return py::make_tuple(states, masks, players, indices);
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
    if (repeats.ndim() != 1 || repeats.shape(0) != size()) {
        throw std::invalid_argument("repeats must have one entry per environment");
    }
    const int32_t* rep = repeats.data();
    std::vector<RummyEnv> copies;
    size_t total = 0;
    for (int i = 0; i < size(); i++) total += std::max(0, rep[i]);
    copies.reserve(total);
    std::vector<uint8_t> alive_copy;
    alive_copy.reserve(total);
    for (int i = 0; i < size(); i++) {
        for (int r = 0; r < rep[i]; r++) {
            copies.push_back(envs[i]);
            alive_copy.push_back(alive[i]);
        }
    }
    std::unique_ptr<EnvBatch> out(new EnvBatch(std::move(copies), pool ? pool->size() : 1));
    out->alive = alive_copy;
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
        .def_property_readonly("num_envs", &VectorizedRummyEnv::size)
        .def("get", &VectorizedRummyEnv::get, "Copy of live game i.")
        .def("current_players", &VectorizedRummyEnv::current_players, "Player to move in each game (1 or 2).")
        .def("opponent_hands", &VectorizedRummyEnv::opponent_hands,
             "[N,52] bool: cards held by the player not to move (training target, not observable).")
        .def("reset", &VectorizedRummyEnv::reset, "Returns (states[N,obs], masks[N,105] bool).")
        .def("step", &VectorizedRummyEnv::step,
             "Returns (states, masks, rewards[N] float32, dones[N] bool, round_ends[N] bool, "
             "round_outs[N] int32: player who went out, 0 otherwise); finished games are auto-reset.");

    py::class_<EnvBatch>(m, "EnvBatch")
        .def(py::init<const std::vector<RummyEnv>&, int>(), py::arg("envs"), py::arg("num_threads") = 1)
        .def_property_readonly("size", &EnvBatch::size)
        .def("get", &EnvBatch::get, "Copy of game i.")
        .def("alive", &EnvBatch::alive_mask)
        .def("observe", &EnvBatch::observe, "Returns (states, masks, current_players[N] int32).")
        .def("observe_alive", &EnvBatch::observe_alive,
             "Live games only: (states[K,obs], masks[K,105], current_players[K], indices[K]).")
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
