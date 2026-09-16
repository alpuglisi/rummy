#pragma once

#include <vector>
#include <array>
#include <random>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

const int DECK_SIZE = 52;
const int HAND_SIZE = 10;
const int ACTION_SPACE_SIZE = 105; // 1 (Deck) + 52 (Discard Draws) + 52 (Discards)
const int OBS_SPACE_SIZE = DECK_SIZE * 3 + 3; // Hand, Discard Presence, Discard Order, Meta

typedef std::vector<int> Meld;

struct GameState {
    std::array<int8_t, DECK_SIZE> card_locations;
    // 0 = Deck, 1 = P1 Hand, 2 = P2 Hand, 3 = Discard Pile, 4 = Melded/Board

    std::vector<int> discard_pile; // Index 0 is oldest, back() is top card

    int current_player;
    int required_meld_card;
    bool turn_phase_is_discard;
    bool is_terminal;

    float p1_score;
    float p2_score;

    std::vector<int> cached_required_meld; // Caches the meld computed by compute_legal_mask()
};

class RummyEnv {
private:
    GameState state;
    std::mt19937 rng;
    std::vector<int> deck_order;
    int deck_index;

    std::vector<float> observation_buffer;

    void deal_initial_hands();
    void update_observation_buffer();

    // Meld Math
    int get_suit(int card) const { return card / 13; }
    int get_rank(int card) const { return card % 13; }
    int get_point_value(int card) const;
    std::vector<Meld> find_all_possible_melds(const std::vector<int>& hand) const;
    Meld find_largest_meld_with_card(const std::vector<int>& hand, int card) const;
    std::vector<int> get_hand(int player) const;
    void auto_meld(int player);

    // Action validation (Not const because they mutate the cache)
    std::vector<uint8_t> compute_legal_mask();
    bool is_legal_action(int action);

    // Resolves the deep-draw meld obligation.
    bool resolve_meld(int player, int discard_card);
    float settle_terminal(int winner);

public:
    RummyEnv(uint32_t seed);

    void reset();
    py::tuple step(int action);

    bool is_done() const { return state.is_terminal; }
    float get_score(int player) const;

    py::array_t<uint8_t> get_legal_actions();
    py::array_t<float> get_state() const;
};
