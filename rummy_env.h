#pragma once

#include <vector>
#include <array>
#include <random>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <exception>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

const int DECK_SIZE = 52;
const int HAND_SIZE = 7;
const int ACTION_SPACE_SIZE = 105; // 1 (Deck) + 52 (Discard Draws) + 52 (Discards)
// Channels: hand, discard presence, discard order, melded board, opponent's
// publicly known cards, unseen cards (still in the deck or hidden in the
// opponent's hand); scalars: own hand size, opponent hand size, score
// difference, then the original three meta flags (turn phase, deck fraction,
// required meld) which stay at the end since the trainer reads obs[-3].
const int OBS_SPACE_SIZE = DECK_SIZE * 6 + 6;

typedef std::vector<int> Meld;

struct GameState {
    std::array<int8_t, DECK_SIZE> card_locations;
    std::array<uint8_t, DECK_SIZE> publicly_known; // Taken from the pile in view of both players
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
    int get_current_player() const { return state.current_player; }

    // Redeal the cards the current player cannot see (opponent's unknown hand
    // cards and the undrawn deck) at random. The current player's observation
    // is unchanged, so search over clones never uses hidden information.
    void randomize_hidden(uint32_t seed);

    py::array_t<uint8_t> get_legal_actions();
    py::array_t<float> get_state() const;

    const std::vector<float>& observation() const { return observation_buffer; }
    std::vector<uint8_t> legal_mask() { return compute_legal_mask(); }
};

// Persistent worker threads. run(fn) calls fn(worker_index, num_workers) on
// every worker and blocks until all have finished; per-step thread creation
// was more expensive than the work itself for a few hundred envs.
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable start_cv, done_cv;
    std::function<void(int, int)> task;
    uint64_t generation = 0;
    int pending = 0;
    bool stopping = false;
    std::exception_ptr error;

    void worker_loop(int index);

public:
    explicit ThreadPool(int num_workers);
    ~ThreadPool();
    int size() const { return static_cast<int>(workers.size()); }
    void run(const std::function<void(int, int)>& fn);
};

// Run fn(i) for i in [0, n) across the pool, or inline when there is no pool.
template <typename F>
void parallel_for(ThreadPool* pool, int n, F&& fn) {
    if (!pool || pool->size() <= 1 || n < 2) {
        for (int i = 0; i < n; i++) fn(i);
        return;
    }
    pool->run([&](int k, int t) {
        for (int i = k; i < n; i += t) fn(i);
    });
}

void write_observation(const RummyEnv& env, int i, float* states, bool* masks);

// Steps N independent engines in one call with the GIL released, optionally
// across a fixed number of threads. Terminal games are reset automatically and
// the returned state/mask are from the fresh game.
class VectorizedRummyEnv {
private:
    std::vector<RummyEnv> envs;
    std::unique_ptr<ThreadPool> pool;

public:
    VectorizedRummyEnv(int num_envs, uint32_t seed, int num_threads);

    int size() const { return static_cast<int>(envs.size()); }
    RummyEnv get(int i) const { return envs.at(i); }
    py::tuple reset();
    py::tuple step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions);
};

// A batch of arbitrary games (copied from Python RummyEnv objects) stepped in
// parallel without auto-reset: finished games keep their terminal state and
// ignore further actions. Used for evaluation matches and search rollouts.
class EnvBatch {
private:
    std::vector<RummyEnv> envs;
    std::vector<uint8_t> alive;
    std::unique_ptr<ThreadPool> pool;

public:
    EnvBatch(const std::vector<RummyEnv>& sources, int num_threads);

    int size() const { return static_cast<int>(envs.size()); }
    RummyEnv get(int i) const { return envs.at(i); }
    py::array_t<bool> alive_mask() const;
    py::tuple observe();
    py::tuple step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions);
    py::array_t<float> scores() const;
    void randomize_hidden(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds);
};
