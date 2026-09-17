#pragma once

#include <vector>
#include <array>
#include <random>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <exception>
#include <cmath>
#include <limits>
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
const int DEFAULT_HAND_SIZE = 7;
const int ACTION_SPACE_SIZE = 105; // 1 (Deck) + 52 (Discard Draws) + 52 (Discards)
// Turn history in the observation: the last HISTORY_LEN draw/discard events
// of the round, oldest first, each encoded as [by opponent, kind one-hot
// (deck draw, pile take, discard), rank one-hot (13), suit one-hot (4),
// cards taken / 10]. Empty slots are all zero.
const int HISTORY_LEN = 12;
const int EVENT_DIM = 1 + 3 + 13 + 4 + 1;
const int HISTORY_SIZE = HISTORY_LEN * EVENT_DIM;
// Channels: hand, discard presence, discard order, melded board, opponent's
// publicly known cards, unseen cards (still in the deck or hidden in the
// opponent's hand); then the turn history; scalars: own score / target,
// opponent score / target, own hand size, opponent hand size, score
// difference, then the original three meta flags (turn phase, deck fraction,
// required meld) which stay at the end since the trainer reads obs[-3].
const int OBS_SPACE_SIZE = DECK_SIZE * 6 + HISTORY_SIZE + 8;
// Auxiliary training targets (ground truth the player cannot see), from the
// player to move's perspective: opponent's hand (52), table cards whose meld
// the opponent can extend (52), own hand cards the opponent could take if
// discarded (52), points the opponent would score at once from each such
// discard / 50 (52), then opponent holds a complete meld, opponent hand
// points / 100, opponent can go out next turn, one of the next three deck
// cards would complete a meld for the player to move.
const int AUX_SPACE_SIZE = DECK_SIZE * 4 + 4;
const int DEFAULT_TARGET_SCORE = 500;
// Safety net so a game between two players who never score cannot run
// forever: after this many rounds the higher score wins.
const int DEFAULT_MAX_ROUNDS = 100;

typedef std::vector<int> Meld;

struct Event {
    int8_t player;
    int8_t kind;    // 0 deck draw, 1 pile take, 2 discard
    int8_t card;    // discarded card, or deepest card taken; -1 for a deck draw
    int8_t count;   // cards taken from the pile
};

struct GameState {
    std::array<int8_t, DECK_SIZE> card_locations;
    std::array<uint8_t, DECK_SIZE> publicly_known; // Taken from the pile in view of both players
    // 0 = Deck, 1 = P1 Hand, 2 = P2 Hand, 3 = Discard Pile, 4 = Melded/Board

    std::vector<int> discard_pile; // Index 0 is oldest, back() is top card
    std::vector<Meld> table;       // Melds on the board, in the order they were laid; lay-offs extend them
    std::vector<Event> history;    // Draw/discard events this round, oldest first

    int current_player;
    int required_meld_card;
    bool turn_phase_is_discard;
    bool is_terminal;

    float p1_score;
    float p2_score;
    float round_start_p1;   // Scores when the current round was dealt
    float round_start_p2;
    int round_number;
    bool round_just_ended;  // The last step ended a round (a new one has been dealt unless the game is over)
    int last_round_out;     // Player who went out to end the last round, 0 if the deck ran out
    int penalised_player;   // Seat that broke the pile-draw obligation (0 = none); ends the game

    std::vector<int> cached_required_meld; // Caches the meld computed by compute_legal_mask()
};

class RummyEnv {
private:
    GameState state;
    std::mt19937 rng;
    std::vector<int> deck_order;
    int deck_index;

    std::vector<float> observation_buffer;
    int manual_meld_player = 0;   // Seat that chooses its own melds (0 = none: everyone auto-melds)
    int target_score;             // The game ends after the round in which a player reaches this
    int max_rounds;               // ...or after this many rounds, whichever comes first
    int hand_size;                // Cards dealt to each player at the start of a round

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
    // Legality of taking the pile from depth `index` for `player` (the rule
    // behind the draw-phase mask), and what the deepest card would score.
    bool can_take_pile(int player, size_t index) const;
    int immediate_points(const std::vector<int>& hand, int card) const;
    // Would `player`, after adding `extra` cards, be able to go out this turn
    // (auto-melding leaves at most one card to discard)?
    bool could_go_out(int player, const std::vector<int>& extra) const;

    // Board bookkeeping. place_meld moves a fresh meld from hand to the table;
    // lay_off_card adds one hand card to table meld `index`; find_lay_off
    // returns the first table meld the card extends, or -1.
    void place_meld(int player, const Meld& m);
    void lay_off_card(int player, int card, int index);
    int find_lay_off(int card) const;

    // Action validation (Not const because they mutate the cache)
    std::vector<uint8_t> compute_legal_mask();
    bool is_legal_action(int action);

    // Resolves the deep-draw meld obligation. Fails (returns false) if
    // discard_card is itself a member of the required meld -- discarding
    // a meld card breaks the meld, per game rules, and the caller is
    // expected to apply the -50 penalty in that case.
    bool resolve_meld(int player, int discard_card);

    // A game is a sequence of rounds. start_round() reshuffles and deals with
    // the scores kept; end_round() subtracts each player's leftover hand from
    // their own score, ends the game if a player has reached target_score,
    // and otherwise deals the next round. Returns the acting player's reward:
    // their score gain over the round minus the opponent's, plus +/-100 when
    // the game ends with a winner.
    void start_round();
    float end_round(int acting_player);

public:
    RummyEnv(uint32_t seed, int target_score = DEFAULT_TARGET_SCORE, int max_rounds = DEFAULT_MAX_ROUNDS,
             int hand_size = DEFAULT_HAND_SIZE);

    void reset();
    py::tuple step(int action);
    std::pair<float, bool> step_raw(int action);

    bool is_done() const { return state.is_terminal; }
    float get_score(int player) const;
    int get_current_player() const { return state.current_player; }
    int get_round() const { return state.round_number; }
    bool round_ended() const { return state.round_just_ended; }
    int get_last_round_out() const { return state.last_round_out; }
    int get_hand_size() const { return hand_size; }
    // (player, kind, card, count) for each event of this round, oldest first.
    std::vector<std::array<int, 4>> get_history() const;
    int get_target_score() const { return target_score; }
    int get_penalised() const { return state.penalised_player; }
    // Card taken from the pile that must be played this turn, or -1.
    int get_required_card() const { return state.required_meld_card; }

    // Redeal the cards the current player cannot see (opponent's unknown hand
    // cards and the undrawn deck) at random. The current player's observation
    // is unchanged, so search over clones never uses hidden information.
    void randomize_hidden(uint32_t seed);

    // Manual melding for a human seat. Auto-meld is skipped for that player;
    // during their discard phase they may lay down valid sets/runs with
    // meld() and extend table melds with lay_off(). The card taken from the
    // pile must be played (melded or laid off) before anything else. A player
    // must always keep one card to discard: going out happens only by
    // discarding the last card.
    void set_manual_meld(int player) { manual_meld_player = player; }
    int get_manual_meld() const { return manual_meld_player; }
    bool is_valid_meld(const std::vector<int>& cards) const;
    py::tuple meld(const std::vector<int>& cards);
    bool can_lay_off(int card, const Meld& meld) const;
    py::tuple lay_off(int card, int meld_index);
    std::vector<Meld> get_table() const { return state.table; }

    // As randomize_hidden, but the opponent's unknown cards are drawn from the
    // hidden pool in proportion to weights[card] (a belief that the card is in
    // their hand); the rest of the pool becomes the deck in random order.
    void randomize_hidden_weighted(uint32_t seed, const float* weights);
    void randomize_hidden_weighted_py(uint32_t seed,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> weights);

    py::array_t<uint8_t> get_legal_actions();
    py::array_t<float> get_state() const;

    const std::vector<float>& observation() const { return observation_buffer; }
    std::vector<uint8_t> legal_mask() { return compute_legal_mask(); }

    // Ground truth for the auxiliary prediction target: the cards held by the
    // player who is not to move. Training-time only; never part of the observation.
    void opponent_hand(bool* out) const;
    py::array_t<bool> get_opponent_hand() const;
    // Auxiliary targets (AUX_SPACE_SIZE floats, see the constant's comment).
    void aux_targets(float* out) const;
    py::array_t<float> get_aux_targets() const;
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
    VectorizedRummyEnv(int num_envs, uint32_t seed, int num_threads, int target_score = DEFAULT_TARGET_SCORE,
                       int hand_size = DEFAULT_HAND_SIZE);

    int size() const { return static_cast<int>(envs.size()); }
    RummyEnv get(int i) const { return envs.at(i); }
    py::array_t<int32_t> current_players() const;
    py::array_t<bool> opponent_hands() const;
    py::array_t<float> aux_targets();
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
    EnvBatch(std::vector<RummyEnv>&& sources, int num_threads);

    // Build the rollout batch for a determinized search in one call: for each
    // source game, `worlds` copies with the hidden cards redealt from seeds[i, w]
    // (weighted by weights[i] when given), each repeated repeats[i] times, one
    // per candidate action. Order: source, world, candidate. Every copy
    // auto-melds for both seats.
    static std::unique_ptr<EnvBatch> for_search(
        const std::vector<RummyEnv>& sources,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> repeats,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds,
        py::object weights, int num_threads);

    int size() const { return static_cast<int>(envs.size()); }
    RummyEnv get(int i) const { return envs.at(i); }
    py::array_t<bool> alive_mask() const;
    py::tuple observe();
    // Observation for the live games only: (states, masks, players, indices).
    py::tuple observe_alive();
    py::tuple step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions);
    py::array_t<float> scores() const;
    py::array_t<int32_t> penalised() const;
    // Stop stepping the flagged games (e.g. once their first round is over).
    void halt(py::array_t<bool, py::array::c_style | py::array::forcecast> which);
    void randomize_hidden(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds);
    void randomize_hidden_weighted(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> seeds,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> weights);
};
