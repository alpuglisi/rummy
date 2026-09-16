#pragma once

#include <vector>
#include <array>
#include <random>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
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

    // Resolves the deep-draw meld obligation. Fails (returns false) if
    // discard_card is itself a member of the required meld -- discarding
    // a meld card breaks the meld, per game rules, and the caller is
    // expected to apply the -50 penalty in that case.
    bool resolve_meld(int player, int discard_card);
    float settle_terminal(int winner);

public:
    RummyEnv(uint32_t seed);

    void reset();
    py::tuple step(int action);
    std::pair<float, bool> step_raw(int action);

    bool is_done() const { return state.is_terminal; }
    float get_score(int player) const;

    py::array_t<uint8_t> get_legal_actions();
    py::array_t<float> get_state() const;

    const std::vector<float>& observation() const { return observation_buffer; }
    std::vector<uint8_t> legal_mask() { return compute_legal_mask(); }
};

// Steps N independent engines in one call with the GIL released, optionally
// across a fixed number of threads. Terminal games are reset automatically and
// the returned state/mask are from the fresh game.
class VectorizedRummyEnv {
private:
    std::vector<RummyEnv> envs;
    int num_threads;

    template <typename F>
    void for_each_env(F&& fn) {
        const int n = static_cast<int>(envs.size());
        const int t = std::min(num_threads, n);
        if (t <= 1) {
            for (int i = 0; i < n; i++) fn(i);
            return;
        }
        std::vector<std::thread> threads;
        std::exception_ptr error;
        std::mutex error_mutex;
        for (int k = 0; k < t; k++) {
            threads.emplace_back([&, k]() {
                try {
                    for (int i = k; i < n; i += t) fn(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                }
            });
        }
        for (auto& th : threads) th.join();
        if (error) std::rethrow_exception(error);
    }

    void write_obs(int i, float* states, bool* masks);

public:
    VectorizedRummyEnv(int num_envs, uint32_t seed, int num_threads);

    int size() const { return static_cast<int>(envs.size()); }
    py::tuple reset();
    py::tuple step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions);
};
