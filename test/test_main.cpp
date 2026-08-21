// Minimal test harness: CHECK records failures and keeps going.

#include "game/gomoku.h"
#include "mcts/mcts.h"
#include "train/evaluator.h"
#include "train/model_expand.h"
#include "train/replay_buffer.h"
#include "train/self_play.h"
#include "train/trainer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <limits>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                        \
  } while (0)

namespace {

using az::Gomoku;

// Plays actions in order; returns false early if any move was illegal.
bool PlayMoves(Gomoku &game, std::initializer_list<int> actions) {
  for (int action : actions) {
    if (!game.Apply(action)) {
      return false;
    }
  }
  return true;
}

int A(int row, int column) { return row * Gomoku::kBoardSize + column; }

void TestHorizontalWin() {
  Gomoku game;
  // Black: (7,3)..(7,7), White plays elsewhere.
  CHECK(PlayMoves(game, {A(7, 3), A(0, 0), A(7, 4), A(0, 1), A(7, 5), A(0, 2),
                         A(7, 6), A(0, 3), A(7, 7)}));
  CHECK(game.IsTerminal());
  CHECK(game.Result() == Gomoku::kBlack);
  CHECK(game.current_player() == Gomoku::kBlack); // no switch after terminal
}

void TestVerticalAndDiagonalWin() {
  {
    Gomoku game;
    CHECK(PlayMoves(game, {A(2, 5), A(13, 0), A(3, 5), A(13, 1), A(4, 5),
                           A(13, 2), A(5, 5), A(13, 3), A(6, 5)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
  {
    Gomoku game;
    // Black diagonal (1,1)(2,2)(3,3)(4,4)(5,5); white on row 13.
    CHECK(PlayMoves(game, {A(1, 1), A(13, 0), A(2, 2), A(13, 1), A(3, 3),
                           A(13, 2), A(4, 4), A(13, 3), A(5, 5)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
  {
    Gomoku game;
    // Black anti-diagonal: (1,10)(2,9)(3,8)(4,7)(5,6); white on row 13.
    CHECK(PlayMoves(game, {A(1, 10), A(13, 0), A(2, 9), A(13, 1), A(3, 8),
                           A(13, 2), A(4, 7), A(13, 3), A(5, 6)}));
    CHECK(game.Result() == Gomoku::kBlack);
  }
}

void TestIllegalMoves() {
  Gomoku game;
  CHECK(game.Apply(A(7, 7)));
  CHECK(!game.Apply(A(7, 7)));       // occupied
  CHECK(!game.Apply(-1));            // out of range
  CHECK(!game.Apply(Gomoku::kCellNum)); // out of range
  CHECK(game.move_count() == 1);
}

void TestDraw() {
  Gomoku game;
  // Full-board pattern with no five-in-a-row for either color and exactly
  // 113 black / 112 white cells so a fully alternating playout fits.
  // Row cell (r,c) is black iff ((c + 2r + s_r) mod 8) < 4 with the
  // per-row phase offsets below (found by exhaustive backtracking over
  // phases; no_five below re-verifies the result independently).
  static const int kRowShift[15] = {0, 6, 4, 4, 6, 6, 4, 4,
                                    6, 7, 4, 6, 1, 7, 5};
  int black_actions[Gomoku::kCellNum];
  int white_actions[Gomoku::kCellNum];
  int black_num = 0, white_num = 0;
  for (int row = 0; row < Gomoku::kBoardSize; ++row) {
    const int shift = kRowShift[row];
    for (int column = 0; column < Gomoku::kBoardSize; ++column) {
      const bool is_black = ((column + 2 * row + shift) % 8) < 4;
      if (is_black) {
        black_actions[black_num++] = A(row, column);
      } else {
        white_actions[white_num++] = A(row, column);
      }
    }
  }
  // Check pattern truly has no 5-in-a-row before playing it out.
  auto no_five = [](const int *cells, int num) {
    std::array<int8_t, Gomoku::kCellNum> filled{};
    for (int i = 0; i < num; ++i) filled[cells[i]] = 1;
    static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int r = 0; r < Gomoku::kBoardSize; ++r) {
      for (int c = 0; c < Gomoku::kBoardSize; ++c) {
        if (!filled[r * Gomoku::kBoardSize + c]) continue;
        for (const auto &d : kDirs) {
          int count = 1;
          for (int s = 1; s < 5; ++s) {
            const int rr = r + d[0] * s, cc = c + d[1] * s;
            if (rr < 0 || rr >= Gomoku::kBoardSize || cc < 0 ||
                cc >= Gomoku::kBoardSize ||
                !filled[rr * Gomoku::kBoardSize + cc])
              break;
            ++count;
          }
          if (count >= 5) return false;
        }
      }
    }
    return true;
  };
  CHECK(no_five(black_actions, black_num));
  CHECK(no_five(white_actions, white_num));
  CHECK(black_num == (Gomoku::kCellNum + 1) / 2);
  CHECK(white_num == Gomoku::kCellNum / 2);

  // Interleave moves by move number: black first (113 blacks, 112 whites).
  bool ok = true;
  for (int move = 0; move < Gomoku::kCellNum; ++move) {
    const int action = (move % 2 == 0) ? black_actions[move / 2]
                                       : white_actions[move / 2];
    if (!game.Apply(action)) {
      std::printf("draw playout broke at move %d action %d\n", move, action);
      ok = false;
      break;
    }
  }
  CHECK(ok);
  CHECK(game.move_count() == Gomoku::kCellNum);
  CHECK(game.Result() == 2); // draw
}

void TestEncode() {
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)})); // black, then white
  deeplearning::FloatTensor4D tensor;
  game.Encode(tensor);
  CHECK(tensor.batch() == 1 && tensor.channels() == Gomoku::kPlaneNum);
  CHECK(tensor.height() == 15 && tensor.width() == 15);
  // Black to move: own = black stones, opponent = white stones.
  CHECK(tensor(0, 0, 7, 7) == 1.0f); // own stone
  CHECK(tensor(0, 1, 8, 8) == 1.0f); // opponent stone
  CHECK(tensor(0, 2, 8, 8) == 1.0f); // last move marker
  float side_sum = 0.0f;
  for (int i = 0; i < Gomoku::kCellNum; ++i) {
    side_sum += tensor.values()[3 * Gomoku::kCellNum + i];
  }
  CHECK(side_sum == static_cast<float>(Gomoku::kCellNum)); // black to move
  float own_sum = 0.0f;
  for (int i = 0; i < Gomoku::kCellNum; ++i) {
    own_sum += tensor.values()[i];
  }
  CHECK(own_sum == 1.0f);

  Gomoku restored;
  restored.MutableBoard()[A(7, 7)] = Gomoku::kBlack;
  restored.MutableBoard()[A(8, 8)] = Gomoku::kWhite;
  restored.SetState(Gomoku::kBlack, 2, 0, A(8, 8));
  restored.Encode(tensor);
  CHECK(tensor(0, 2, 8, 8) == 1.0f);
}

void TestEvalCacheIncludesLastMove() {
  Gomoku first;
  first.MutableBoard()[A(7, 7)] = Gomoku::kBlack;
  first.MutableBoard()[A(8, 8)] = Gomoku::kWhite;
  first.SetState(Gomoku::kBlack, 2, 0, A(7, 7));
  Gomoku second = first;
  second.SetState(Gomoku::kBlack, 2, 0, A(8, 8));

  az::EvalCache cache;
  float policy[Gomoku::kActionNum] = {};
  policy[A(6, 6)] = 1.0f;
  cache.Store(first, policy, 0.25f);
  az::EvalCache::Entry entry;
  CHECK(cache.Lookup(first, entry));
  CHECK(std::fabs(entry.value - 0.25f) < 1e-6f);
  CHECK(!cache.Lookup(second, entry));
}

void TestSymmetryConsistency() {
  using az::Gomoku;
  // All 8 transforms are bijections on cells.
  for (int s = 0; s < Gomoku::kSymmetryNum; ++s) {
    std::array<bool, Gomoku::kCellNum> seen{};
    for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
      const int mapped = Gomoku::TransformAction(cell, s);
      CHECK(mapped >= 0 && mapped < Gomoku::kCellNum);
      seen[mapped] = true;
    }
    for (bool found : seen) CHECK(found);
  }
  // Transform twice with identity-symmetric op pairs recovers original:
  // rotation 180 twice = identity (s=4 -> rotate180 twice).
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    CHECK(Gomoku::TransformAction(Gomoku::TransformAction(cell, 4), 4) ==
          cell);
    CHECK(Gomoku::TransformAction(Gomoku::TransformAction(cell, 1), 1) ==
          cell); // mirror twice = identity
  }
  // Planes/policies transform identically with a delta input.
  Gomoku game;
  CHECK(PlayMoves(game, {A(2, 3), A(5, 6), A(9, 1)}));
  deeplearning::FloatTensor4D tensor;
  game.Encode(tensor);
  for (int s = 0; s < Gomoku::kSymmetryNum; ++s) {
    std::vector<float> planes_out(Gomoku::kPlaneNum * Gomoku::kCellNum);
    Gomoku::TransformPlanes(tensor.data(), planes_out.data(), s);
    // own-stone count must be preserved under any symmetry.
    float own_sum = 0.0f;
    for (int i = 0; i < Gomoku::kCellNum; ++i) own_sum += planes_out[i];
    float own_expected = 0.0f;
    for (int i = 0; i < Gomoku::kCellNum; ++i) own_expected += tensor.values()[i];
    CHECK(std::fabs(own_sum - own_expected) < 1e-4f);
  }
}

} // namespace

// Uniform-prior / zero-value fake evaluator for MCTS mechanics tests.
class FlatEvaluator : public az::INetEvaluator {
public:
  void Predict(const az::Gomoku &game, float *policy, float &value) override {
    int legal = 0;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      if (game.IsLegal(a)) ++legal;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
    value = 0.0f;
  }
};

class CountingEvaluator : public az::INetEvaluator {
public:
  int calls = 0;
  void Predict(const az::Gomoku &game, float *policy, float &value) override {
    ++calls;
    int legal = 0;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      if (game.IsLegal(a)) ++legal;
    for (int a = 0; a < az::Gomoku::kActionNum; ++a)
      policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
    value = 0.0f;
  }
};

namespace {

void TestMctsForcedWin() {
  using az::Gomoku;
  // Black has an open four at (7,3..6); winning replies: (7,2) and (7,7).
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 3), A(0, 0), A(7, 4), A(0, 1), A(7, 5), A(0, 2),
                         A(7, 6), A(0, 3)}));
  CHECK(!game.IsTerminal());
  CHECK(game.current_player() == Gomoku::kBlack);

  FlatEvaluator evaluator;
  az::MctsConfig config;
  config.simulation_num_ = 300;
  config.dirichlet_epsilon_ = 0.0f;
  az::Mcts mcts;
  std::mt19937 rng(1);
  std::vector<int> visit_action, visit_count;
  mcts.Search(game, config, evaluator, rng, visit_action, visit_count);

  int best = -1, best_count = -1;
  for (std::size_t i = 0; i < visit_action.size(); ++i) {
    if (visit_count[i] > best_count) {
      best_count = visit_count[i];
      best = visit_action[i];
    }
  }
  // MCTS must have found the immediate win.
  CHECK(best == A(7, 2) || best == A(7, 7));
}

void TestVisitDistribution() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  FlatEvaluator evaluator;
  az::MctsConfig config;
  config.simulation_num_ = 60;
  config.dirichlet_epsilon_ = 0.0f;
  az::Mcts mcts;
  std::mt19937 rng(2);
  std::vector<int> visit_action, visit_count;
  mcts.Search(game, config, evaluator, rng, visit_action, visit_count);

  std::vector<float> pi(Gomoku::kActionNum, 0.0f);
  az::Mcts::VisitDistribution(visit_action, visit_count, pi.data());
  float sum = 0.0f;
  for (float p : pi) sum += p;
  CHECK(std::fabs(sum - 1.0f) < 1e-4f || sum == 0.0f);
  CHECK(pi[A(7, 7)] == 0.0f && pi[A(8, 8)] == 0.0f); // occupied cells get none
}

void TestMctsTreeReuse() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 80;
  config.dirichlet_epsilon_ = 0.0f;
  config.reuse_tree_ = true;
  config.max_retained_nodes_ = 1000;
  config.max_retained_edges_ = 10000;
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(11);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());

  int best = -1, best_count = -1;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (visits[i] > best_count) {
      best = actions[i];
      best_count = visits[i];
    }
  }
  CHECK(best >= 0 && game.IsLegal(best));
  CHECK(mcts.AdvanceRoot(best));
  CHECK(game.Apply(best));
  CHECK(mcts.node_count() <=
        static_cast<std::size_t>(config.max_retained_nodes_));
  CHECK(mcts.edge_count() <=
        static_cast<std::size_t>(config.max_retained_edges_));
  const int inherited_visits = mcts.root_visits();
  CHECK(inherited_visits > 0);

  evaluator.calls = 0;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(mcts.last_search_reused());
  CHECK(mcts.root_visits() >= inherited_visits + config.simulation_num_);
  const int reused_calls = evaluator.calls;

  CountingEvaluator fresh_evaluator;
  az::Mcts fresh;
  fresh.Search(game, config, fresh_evaluator, rng, actions, visits);
  CHECK(!fresh.last_search_reused());
  CHECK(reused_calls < fresh_evaluator.calls);

  Gomoku unrelated;
  CHECK(PlayMoves(unrelated, {A(0, 0), A(14, 14)}));
  mcts.Search(unrelated, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());
}

void TestMctsReuseDisabledByDefault() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 20;
  config.dirichlet_epsilon_ = 0.0f;
  CHECK(!config.reuse_tree_);
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(17);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(mcts.root_visits() == 20);
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.last_search_reused());
  CHECK(mcts.root_visits() == 20); // did not accumulate across calls
}

void TestMctsReuseSafetyAndBounds() {
  using az::Gomoku;
  Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 40;
  config.dirichlet_epsilon_ = 0.0f;
  config.reuse_tree_ = true;
  config.max_retained_nodes_ = 500;
  config.max_retained_edges_ = 5000;
  CountingEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(29);
  std::vector<int> actions, visits;

  // Public AdvanceRoot must reject malformed external input without indexing
  // the board out of range.
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.AdvanceRoot(-1));
  CHECK(mcts.node_count() == 0);
  mcts.Search(game, config, evaluator, rng, actions, visits);
  CHECK(!mcts.AdvanceRoot(Gomoku::kActionNum));
  CHECK(mcts.node_count() == 0);

  // Exercise many root advances. After every real move the retained tree is
  // either within both hard budgets or was dropped completely for rebuild.
  for (int move = 0; move < 50 && !game.IsTerminal(); ++move) {
    mcts.Search(game, config, evaluator, rng, actions, visits);
    int best = -1, best_count = -1;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (visits[i] > best_count) {
        best = actions[i];
        best_count = visits[i];
      }
    }
    CHECK(best >= 0 && game.IsLegal(best));
    const bool retained = mcts.AdvanceRoot(best);
    CHECK(game.Apply(best));
    if (retained) {
      CHECK(mcts.node_count() <=
            static_cast<std::size_t>(config.max_retained_nodes_));
      CHECK(mcts.edge_count() <=
            static_cast<std::size_t>(config.max_retained_edges_));
    } else {
      CHECK(mcts.node_count() == 0);
      CHECK(mcts.edge_count() == 0);
    }
  }

  az::MctsConfig tight = config;
  tight.simulation_num_ = 80;
  tight.max_retained_nodes_ = 10;
  tight.max_retained_edges_ = 5000;
  az::Mcts bounded;
  Gomoku bounded_game;
  CHECK(PlayMoves(bounded_game, {A(7, 7), A(8, 8)}));
  bounded.Search(bounded_game, tight, evaluator, rng, actions, visits);
  CHECK(bounded.budget_exhausted());
  CHECK(bounded.node_count() <=
        static_cast<std::size_t>(tight.max_retained_nodes_));
  CHECK(bounded.edge_count() <=
        static_cast<std::size_t>(tight.max_retained_edges_));

  az::MctsConfig edge_tight = config;
  edge_tight.simulation_num_ = 80;
  std::vector<int> initial_candidates;
  bounded_game.CandidateActions(initial_candidates);
  edge_tight.max_retained_nodes_ = 1000;
  edge_tight.max_retained_edges_ =
      static_cast<int>(initial_candidates.size()) + 1;
  az::Mcts edge_bounded;
  edge_bounded.Search(bounded_game, edge_tight, evaluator, rng, actions,
                      visits);
  CHECK(edge_bounded.budget_exhausted());
  CHECK(!actions.empty()); // root policy remains usable
  CHECK(edge_bounded.edge_count() <=
        static_cast<std::size_t>(edge_tight.max_retained_edges_));
  int edge_best = actions[0];
  CHECK(!edge_bounded.AdvanceRoot(edge_best)); // exhausted cache is dropped
  CHECK(edge_bounded.node_count() == 0 && edge_bounded.edge_count() == 0);
}

void TestMctsNoiseResetOnReuse() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 0;
  config.reuse_tree_ = true;
  config.normalized_dirichlet_ = true;
  config.dirichlet_epsilon_ = 0.25f;
  FlatEvaluator evaluator;
  az::Mcts mcts;
  std::vector<int> actions;
  std::vector<int> visits;
  std::vector<float> first;
  std::vector<float> second;
  std::mt19937 rng1(31);
  mcts.Search(game, config, evaluator, rng1, actions, visits);
  mcts.RootPriors(actions, first);
  std::mt19937 rng2(31);
  mcts.Search(game, config, evaluator, rng2, actions, visits);
  mcts.RootPriors(actions, second);
  CHECK(mcts.last_search_reused());
  CHECK(first.size() == second.size());
  for (std::size_t i = 0; i < first.size(); ++i)
    CHECK(std::fabs(first[i] - second[i]) < 1e-6f);
}

void TestDirichletNoiseWeight() {
  az::Gomoku game;
  CHECK(PlayMoves(game, {A(7, 7), A(8, 8)}));
  az::MctsConfig config;
  config.simulation_num_ = 0; // inspect root immediately after expansion
  config.dirichlet_epsilon_ = 0.25f;
  config.dirichlet_alpha_ = 0.3f;
  config.normalized_dirichlet_ = true;
  FlatEvaluator evaluator;
  az::Mcts mcts;
  std::mt19937 rng(23), expected_rng(23);
  std::vector<int> actions, visits;
  mcts.Search(game, config, evaluator, rng, actions, visits);
  std::vector<float> prior;
  mcts.RootPriors(actions, prior);
  CHECK(!prior.empty());

  std::gamma_distribution<float> gamma(config.dirichlet_alpha_, 1.0f);
  std::vector<float> noise(prior.size());
  float sum = 0.0f;
  for (float &value : noise) {
    value = gamma(expected_rng);
    sum += value;
  }
  float prior_sum = 0.0f;
  const float base = 1.0f / prior.size();
  for (std::size_t i = 0; i < prior.size(); ++i) {
    const float expected = 0.75f * base + 0.25f * noise[i] / sum;
    CHECK(std::fabs(prior[i] - expected) < 1e-6f);
    prior_sum += prior[i];
  }
  CHECK(std::fabs(prior_sum - 1.0f) < 1e-5f);
}

void TestReplaySaveReportsWriteFailure() {
  az::ReplayBuffer buffer(2);
  az::Sample sample{};
  buffer.Push(sample);
  // Linux's /dev/full accepts open() but reports ENOSPC while flushing. Save
  // must not claim success and replace a valid replay checkpoint afterward.
  CHECK(!buffer.Save("/dev/full"));
}

void TestReplayLoadFailurePreservesBuffer() {
  az::ReplayBuffer buffer(2);
  az::Sample sample{};
  sample.value = 0.5f;
  buffer.Push(sample);
  const char *path = "/tmp/az_bad_replay.bin";
  FILE *out = std::fopen(path, "wb");
  std::size_t bad_size = 99;
  std::size_t bad_position = 0;
  CHECK(out != nullptr);
  if (out != nullptr) {
    std::fwrite(&bad_size, sizeof(bad_size), 1, out);
    std::fwrite(&bad_position, sizeof(bad_position), 1, out);
    std::fclose(out);
  }
  CHECK(!buffer.Load(path));
  CHECK(buffer.Size() == 1); // rejected header never touched live records
  CHECK(std::fabs(buffer.At(0).value - 0.5f) < 1e-6f);
  std::remove(path);

  out = std::fopen(path, "wb");
  std::size_t one = 1;
  std::size_t position = 1;
  CHECK(out != nullptr);
  if (out != nullptr) {
    std::fwrite(&one, sizeof(one), 1, out);
    std::fwrite(&position, sizeof(position), 1, out);
    std::fclose(out); // payload deliberately absent
  }
  CHECK(!buffer.Load(path));
  CHECK(buffer.Size() == 0); // partial payload is never addressable
  std::remove(path);
}

deeplearning::PolicyValueResNet::Config TinyNetConfig(int seed) {
  deeplearning::PolicyValueResNet::Config config;
  config.input_channels_ = Gomoku::kPlaneNum;
  config.board_height_ = Gomoku::kBoardSize;
  config.board_width_ = Gomoku::kBoardSize;
  config.trunk_channels_ = 4;
  config.residual_block_num_ = 1;
  config.policy_channels_ = 2;
  config.policy_size_ = Gomoku::kActionNum;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 8;
  config.thread_num_ = 1;
  config.rand_seed_ = seed;
  return config;
}

void TestTeacherTargetsDoNotControlBehavior() {
  deeplearning::PolicyValueResNet student, teacher;
  CHECK(student.Init(TinyNetConfig(11)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(teacher.Init(TinyNetConfig(97)) ==
        deeplearning::PolicyValueResNet::SUCCESS);

  az::SelfPlayConfig baseline;
  baseline.worker_num_ = 1;
  baseline.game_num_ = 1;
  baseline.max_moves_ = 6;
  baseline.temperature_move_cutoff_ = 6;
  baseline.seed_ = 123;
  baseline.use_cache_ = false;
  baseline.mcts_.simulation_num_ = 2;
  baseline.mcts_.dirichlet_epsilon_ = 0.0f;

  az::ReplayBuffer behavior_buffer(16), teacher_buffer(16);
  const az::SelfPlayStats behavior_stats =
      az::RunSelfPlay(student, baseline, behavior_buffer);

  az::SelfPlayConfig distilled = baseline;
  distilled.teacher_target_prob_ = 1.0f;
  distilled.teacher_simulations_ = 3;
  const az::SelfPlayStats teacher_stats =
      az::RunSelfPlay(student, distilled, teacher_buffer, &teacher);

  CHECK(behavior_buffer.Size() == teacher_buffer.Size());
  CHECK(teacher_stats.teacher_policy_targets == teacher_buffer.Size());
  CHECK(teacher_stats.student_policy_targets == 0);
  CHECK(behavior_stats.teacher_policy_targets == 0);
  CHECK(behavior_stats.student_policy_targets == behavior_buffer.Size());
  bool policy_differs = false;
  for (std::size_t i = 0; i < behavior_buffer.Size(); ++i) {
    const az::Sample &left = behavior_buffer.At(static_cast<int>(i));
    const az::Sample &right = teacher_buffer.At(static_cast<int>(i));
    CHECK(left.planes == right.planes); // behavior trajectory is unchanged
    CHECK(std::fabs(left.value - right.value) < 1e-6f);
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (std::fabs(left.policy[action] - right.policy[action]) > 1e-6f)
        policy_differs = true;
    }
  }
  CHECK(policy_differs);
}

void TestEvalCacheCountersAreThreadSafe() {
  az::EvalCache cache;
  std::array<float, Gomoku::kActionNum> policy{};
  policy[112] = 1.0f;
  const int worker_num = 8, lookup_per_worker = 1000;
  std::vector<Gomoku> games;
  std::vector<std::size_t> shards;
  for (int action = 0;
       action < Gomoku::kActionNum && games.size() < worker_num; ++action) {
    Gomoku game;
    CHECK(game.Apply(action));
    std::string key(Gomoku::kCellNum + 2, '\0');
    key[0] = static_cast<char>(game.current_player());
    key[1] = static_cast<char>(game.last_action() + 1);
    for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
      key[cell + 2] = game.board()[cell] == 0
          ? '\0'
          : static_cast<char>(game.board()[cell] == game.current_player()
                                  ? 1 : 2);
    }
    const std::size_t shard = std::hash<std::string>{}(key) % 64;
    if (std::find(shards.begin(), shards.end(), shard) != shards.end())
      continue;
    shards.push_back(shard);
    games.push_back(game);
    cache.Store(games.back(), policy.data(), 0.25f);
  }
  CHECK(games.size() == worker_num);
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < worker_num; ++worker) {
    workers.emplace_back([&, worker]() {
      for (int i = 0; i < lookup_per_worker; ++i) {
        az::EvalCache::Entry entry;
        if (!cache.Lookup(games[worker], entry)) failures.fetch_add(1);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  CHECK(failures.load() == 0);
  CHECK(cache.Lookups() ==
        static_cast<std::size_t>(worker_num * lookup_per_worker));
  CHECK(cache.Hits() == cache.Lookups());
}

void TestDynamicBatchEvaluatorMatchesSingleEvaluation() {
  deeplearning::PolicyValueResNet master;
  CHECK(master.Init(TinyNetConfig(211)) ==
        deeplearning::PolicyValueResNet::SUCCESS);

  az::Evaluator reference;
  CHECK(reference.Init(TinyNetConfig(211)));
  CHECK(az::AssignWeights(reference.net(), master));

  az::DynamicBatchEvaluator service;
  az::DynamicBatchEvaluator::Config config;
  config.max_batch_size_ = 8;
  config.max_wait_us_ = 20000;
  config.inference_thread_num_ = 1;
  CHECK(service.Init(master, config));

  const int request_num = 8;
  std::vector<Gomoku> games(request_num);
  for (int index = 0; index < request_num; ++index) {
    CHECK(games[index].Apply(A(7, 7)));
    CHECK(games[index].Apply(A(index / 4, index % 4)));
  }
  std::vector<std::array<float, Gomoku::kActionNum>> batched(request_num);
  std::vector<float> batched_values(request_num, 0.0f);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  for (int index = 0; index < request_num; ++index) {
    workers.emplace_back([&, index]() {
      ready.fetch_add(1);
      while (!go.load()) std::this_thread::yield();
      service.Predict(games[index], batched[index].data(),
                      batched_values[index]);
    });
  }
  while (ready.load() != request_num) std::this_thread::yield();
  go.store(true);
  for (auto &worker : workers) worker.join();
  service.Stop();

  for (int index = 0; index < request_num; ++index) {
    std::array<float, Gomoku::kActionNum> expected{};
    float expected_value = 0.0f;
    reference.Predict(games[index], expected.data(), expected_value);
    CHECK(std::fabs(expected_value - batched_values[index]) < 1e-5f);
    float policy_sum = 0.0f;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      CHECK(std::fabs(expected[action] - batched[index][action]) < 1e-5f);
      policy_sum += batched[index][action];
      if (!games[index].IsLegal(action)) CHECK(batched[index][action] == 0.0f);
    }
    CHECK(std::fabs(policy_sum - 1.0f) < 1e-5f);
  }
  const auto stats = service.GetStats();
  CHECK(stats.requests_ == request_num);
  CHECK(stats.total_batch_size_ == request_num);
  CHECK(stats.forward_calls_ >= 1);
  CHECK(stats.forward_calls_ < request_num);
  CHECK(stats.max_batch_size_ > 1);
}

void TestDynamicBatchEvaluatorFlushesPartialBatch() {
  deeplearning::PolicyValueResNet master;
  CHECK(master.Init(TinyNetConfig(212)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  az::DynamicBatchEvaluator service;
  az::DynamicBatchEvaluator::Config config;
  config.max_batch_size_ = 16;
  config.max_wait_us_ = 1000;
  CHECK(service.Init(master, config));
  Gomoku game;
  std::array<float, Gomoku::kActionNum> policy{};
  float value = 0.0f;
  service.Predict(game, policy.data(), value);
  service.Stop();
  const auto stats = service.GetStats();
  CHECK(stats.requests_ == 1);
  CHECK(stats.forward_calls_ == 1);
  CHECK(stats.total_batch_size_ == 1);
  CHECK(stats.max_batch_size_ == 1);
}

void TestDynamicBatchEvaluatorDrainsOnStop() {
  deeplearning::PolicyValueResNet master;
  CHECK(master.Init(TinyNetConfig(214)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  az::DynamicBatchEvaluator service;
  az::DynamicBatchEvaluator::Config config;
  config.max_batch_size_ = 16;
  config.max_wait_us_ = 50000;
  CHECK(service.Init(master, config));

  const int request_num = 8;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int> completed{0};
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  for (int index = 0; index < request_num; ++index) {
    callers.emplace_back([&, index]() {
      Gomoku game;
      if (!game.Apply(A(7, 7))) failures.fetch_add(1);
      if (index > 0 && !game.Apply(A(0, index - 1))) failures.fetch_add(1);
      std::array<float, Gomoku::kActionNum> policy{};
      float value = 0.0f;
      ready.fetch_add(1);
      while (!go.load()) std::this_thread::yield();
      service.Predict(game, policy.data(), value);
      float sum = 0.0f;
      for (float probability : policy) sum += probability;
      if (std::fabs(sum - 1.0f) < 1e-5f) completed.fetch_add(1);
    });
  }
  while (ready.load() != request_num) std::this_thread::yield();
  go.store(true);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  service.Stop();
  for (auto &caller : callers) caller.join();
  CHECK(failures.load() == 0);
  CHECK(completed.load() == request_num);
  CHECK(service.GetStats().requests_ == request_num);
}

void TestPolicyValueResNetExpansionPreservesInference() {
  auto source_config = TinyNetConfig(301);
  deeplearning::PolicyValueResNet source;
  CHECK(source.Init(source_config) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  auto destination_config = source_config;
  destination_config.trunk_channels_ = 8;
  destination_config.residual_block_num_ = 3;
  destination_config.rand_seed_ = 999;
  deeplearning::PolicyValueResNet destination;
  CHECK(destination.Init(destination_config) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(az::ExpandPolicyValueResNet(destination, source));
  const auto &stem_weight = destination.stem_conv().weight();
  const int stem_kernel = source_config.input_channels_ * 3 * 3;
  CHECK(stem_weight[source_config.trunk_channels_ * stem_kernel] != 0.0f);
  CHECK(stem_weight[source_config.trunk_channels_ * stem_kernel] !=
        stem_weight[(source_config.trunk_channels_ + 1) * stem_kernel]);
  CHECK(destination.blocks()[source_config.residual_block_num_]
            .conv1().weight()[0] != 0.0f);

  deeplearning::FloatTensor4D input(3, Gomoku::kPlaneNum,
                                    Gomoku::kBoardSize, Gomoku::kBoardSize);
  Gomoku games[3];
  CHECK(games[1].Apply(A(7, 7)));
  CHECK(games[2].Apply(A(7, 7)));
  CHECK(games[2].Apply(A(7, 8)));
  for (int index = 0; index < 3; ++index) {
    games[index].EncodeInto(
        input.data() + index * Gomoku::kPlaneNum * Gomoku::kCellNum);
  }
  deeplearning::PolicyValueResNet::Output source_output, destination_output;
  CHECK(source.Forward(input, source_output, false) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(destination.Forward(input, destination_output, false) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(source_output.policy_logits_.size() ==
        destination_output.policy_logits_.size());
  CHECK(source_output.values_.size() == destination_output.values_.size());
  for (std::size_t index = 0; index < source_output.policy_logits_.size();
       ++index) {
    CHECK(std::fabs(source_output.policy_logits_[index] -
                    destination_output.policy_logits_[index]) < 2e-5f);
  }
  for (std::size_t index = 0; index < source_output.values_.size(); ++index) {
    CHECK(std::fabs(source_output.values_[index] -
                    destination_output.values_[index]) < 2e-6f);
  }
  // Added blocks keep independent random conv1 features but zero conv2, so
  // their residual output starts as an exact identity.
  for (int block = source_config.residual_block_num_;
       block < destination_config.residual_block_num_; ++block) {
    for (float value : destination.blocks()[block].conv2().weight())
      CHECK(value == 0.0f);
  }

  // Exact-preserving random added features must give both the policy head and
  // the first new block's second convolution a nonzero first-step gradient.
  destination.ZeroGrad();
  CHECK(destination.Forward(input, destination_output, true) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  std::vector<float> grad_policy(3 * Gomoku::kActionNum, 0.0f);
  std::vector<float> grad_value(3, 0.0f);
  grad_policy[0] = 1.0f;
  grad_policy[Gomoku::kActionNum + 1] = -0.5f;
  grad_policy[2 * Gomoku::kActionNum + 2] = 0.25f;
  deeplearning::FloatTensor4D grad_input;
  CHECK(destination.Backward(grad_policy, grad_value, grad_input) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  bool extra_policy_gradient = false;
  float first_extra_gradient = 0.0f;
  bool distinct_extra_gradient = false;
  const auto &policy_grad = destination.policy_conv().grad_weight();
  for (int output = 0; output < destination_config.policy_channels_; ++output) {
    for (int input_channel = source_config.trunk_channels_;
         input_channel < destination_config.trunk_channels_; ++input_channel) {
      const float gradient = policy_grad[
          output * destination_config.trunk_channels_ + input_channel];
      if (std::fabs(gradient) > 1e-12f) {
        extra_policy_gradient = true;
      }
      if (input_channel == source_config.trunk_channels_) {
        first_extra_gradient = gradient;
      } else if (std::fabs(gradient - first_extra_gradient) > 1e-12f) {
        distinct_extra_gradient = true;
      }
    }
  }
  CHECK(extra_policy_gradient);
  CHECK(distinct_extra_gradient);
  bool new_block_gradient = false;
  for (float value : destination.blocks()[source_config.residual_block_num_]
                         .conv2().grad_weight()) {
    if (std::fabs(value) > 1e-10f) new_block_gradient = true;
  }
  CHECK(new_block_gradient);

  // Apply one tiny conv2 update, then the second backward pass must reach the
  // formerly isolated random conv1 features in the added block.
  auto &new_block = destination.blocks()[source_config.residual_block_num_];
  auto &conv2_weight = new_block.conv2().mutable_weight();
  const auto first_grad = new_block.conv2().grad_weight();
  for (std::size_t index = 0; index < conv2_weight.size(); ++index) {
    conv2_weight[index] -= 1e-3f * first_grad[index];
  }
  destination.ZeroGrad();
  CHECK(destination.Forward(input, destination_output, true) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(destination.Backward(grad_policy, grad_value, grad_input) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  bool second_step_reaches_conv1 = false;
  for (float value : new_block.conv1().grad_weight()) {
    if (std::fabs(value) > 1e-10f) second_step_reaches_conv1 = true;
  }
  CHECK(second_step_reaches_conv1);
}

void TestExpandedTrainerCheckpointResume() {
  const char *dir = "/tmp/az_expanded_resume";
  std::system("rm -rf /tmp/az_expanded_resume");
  CHECK(mkdir(dir, 0755) == 0);
  const std::string source_path = std::string(dir) + "/source.net";
  deeplearning::PolicyValueResNet source;
  CHECK(source.Init(TinyNetConfig(401)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(source.Save(source_path) == deeplearning::PolicyValueResNet::SUCCESS);

  az::TrainConfig config;
  config.net_ = TinyNetConfig(999);
  config.net_.trunk_channels_ = 8;
  config.net_.residual_block_num_ = 2;
  config.run_dir_ = dir;
  config.iterations_ = 1;
  config.resume_ = false;
  config.expand_init_model_ = true;
  config.init_model_ = source_path;
  config.init_best_model_ = source_path;
  config.selfplay_.worker_num_ = 1;
  config.selfplay_.game_num_ = 1;
  config.selfplay_.mcts_.simulation_num_ = 1;
  config.selfplay_.mcts_.dirichlet_epsilon_ = 0.0f;
  config.selfplay_.temperature_move_cutoff_ = 0;
  config.selfplay_.max_moves_ = 4;
  config.train_steps_ = 1;
  config.batch_size_ = 1;
  config.buffer_capacity_ = 16;
  config.hard_fraction_ = 0.0f;
  config.gate_every_ = 100;
  config.save_buffer_every_ = 1;
  az::Trainer trainer;
  CHECK(trainer.Init(config));
  trainer.Run();

  deeplearning::PolicyValueResNet checkpoint;
  CHECK(checkpoint.Load(std::string(dir) + "/checkpoint.latest.1.net") ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(checkpoint.config().trunk_channels_ == 8);
  CHECK(checkpoint.config().residual_block_num_ == 2);

  az::TrainConfig resume = config;
  resume.iterations_ = 2;
  resume.resume_ = true;
  resume.init_model_.clear();
  resume.init_best_model_.clear();
  az::Trainer resumed;
  CHECK(resumed.Init(resume));
  resumed.Run();
  CHECK(access((std::string(dir) + "/checkpoint.latest.2.net").c_str(),
               F_OK) == 0);
  std::system("rm -rf /tmp/az_expanded_resume");
}

void TestFreshTrainerLoadsInitialReplay() {
  const char *dir = "/tmp/az_init_replay";
  std::system("rm -rf /tmp/az_init_replay");
  CHECK(mkdir(dir, 0755) == 0);
  const std::string source_path = std::string(dir) + "/source.net";
  const std::string replay_path = std::string(dir) + "/seed.bin";
  deeplearning::PolicyValueResNet source;
  CHECK(source.Init(TinyNetConfig(501)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(source.Save(source_path) == deeplearning::PolicyValueResNet::SUCCESS);
  az::ReplayBuffer seed(16);
  az::Sample sample;
  sample.planes[0] = 1.0f;
  sample.policy[A(7, 7)] = 1.0f;
  sample.value = 1.0f;
  seed.Push(sample);
  CHECK(seed.Save(replay_path));

  az::TrainConfig config;
  config.net_ = TinyNetConfig(501);
  config.run_dir_ = dir;
  config.iterations_ = 1;
  config.resume_ = false;
  config.init_model_ = source_path;
  config.init_best_model_ = source_path;
  config.init_buffer_ = replay_path;
  config.buffer_capacity_ = 16;
  config.selfplay_.worker_num_ = 1;
  config.selfplay_.game_num_ = 1;
  config.selfplay_.mcts_.simulation_num_ = 1;
  config.selfplay_.mcts_.dirichlet_epsilon_ = 0.0f;
  config.selfplay_.max_moves_ = 1;
  config.selfplay_.temperature_move_cutoff_ = 0;
  config.train_steps_ = 0;
  config.gate_every_ = 100;
  config.save_buffer_every_ = 1;
  az::Trainer trainer;
  CHECK(trainer.Init(config));
  trainer.Run();
  az::ReplayBuffer loaded(16);
  CHECK(loaded.Load(std::string(dir) + "/checkpoint.buffer.1.bin"));
  CHECK(loaded.Size() == 2);
  CHECK(loaded.At(0).planes[0] == 1.0f);
  CHECK(loaded.At(0).value == 1.0f);
  std::system("rm -rf /tmp/az_init_replay");
}

void TestSelfPlayUsesDynamicBatchEvaluator() {
  deeplearning::PolicyValueResNet master;
  CHECK(master.Init(TinyNetConfig(213)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  az::SelfPlayConfig config;
  config.worker_num_ = 4;
  config.game_num_ = 4;
  config.max_moves_ = 4;
  config.temperature_move_cutoff_ = 0;
  config.use_cache_ = false;
  config.mcts_.simulation_num_ = 1;
  config.mcts_.dirichlet_epsilon_ = 0.0f;
  config.use_batch_inference_ = true;
  config.inference_batch_size_ = 4;
  config.inference_wait_us_ = 20000;
  config.inference_thread_num_ = 1;
  az::ReplayBuffer buffer(64);
  const az::SelfPlayStats stats = az::RunSelfPlay(master, config, buffer);
  CHECK(stats.games == 4);
  CHECK(stats.samples == buffer.Size());
  CHECK(stats.samples == 16);
  CHECK(stats.inference_forward_calls > 0);
  CHECK(stats.inference_average_batch > 1.0);
  CHECK(stats.inference_max_batch > 1);
}

void TestExpandedStudentSupportsSmallerTeacher() {
  deeplearning::PolicyValueResNet teacher, student;
  const auto teacher_config = TinyNetConfig(215);
  CHECK(teacher.Init(teacher_config) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  auto student_config = teacher_config;
  student_config.trunk_channels_ = 8;
  student_config.residual_block_num_ = 2;
  CHECK(student.Init(student_config) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(az::ExpandPolicyValueResNet(student, teacher));

  for (int batched = 0; batched <= 1; ++batched) {
    az::SelfPlayConfig config;
    config.worker_num_ = 2;
    config.game_num_ = 2;
    config.max_moves_ = 2;
    config.temperature_move_cutoff_ = 0;
    config.use_cache_ = false;
    config.mcts_.simulation_num_ = 1;
    config.mcts_.dirichlet_epsilon_ = 0.0f;
    config.teacher_target_prob_ = 1.0f;
    config.teacher_simulations_ = 1;
    config.use_batch_inference_ = batched != 0;
    config.inference_batch_size_ = 2;
    config.inference_wait_us_ = 10000;
    config.inference_thread_num_ = 1;
    az::ReplayBuffer buffer(16);
    const auto stats = az::RunSelfPlay(student, config, buffer, &teacher);
    CHECK(stats.games == 2);
    CHECK(stats.samples == 4);
    CHECK(stats.teacher_policy_targets == 4);
    CHECK(stats.student_policy_targets == 0);
    if (batched) CHECK(stats.inference_forward_calls > 0);
  }
}

void TestFastTrainerRejectsInvalidConfigAndMismatchedBest() {
  az::TrainConfig invalid;
  invalid.hard_fraction_ = std::numeric_limits<float>::quiet_NaN();
  az::Trainer invalid_trainer;
  CHECK(!invalid_trainer.Init(invalid));
  az::TrainConfig invalid_threads;
  invalid_threads.train_thread_num_ = 0;
  az::Trainer invalid_thread_trainer;
  CHECK(!invalid_thread_trainer.Init(invalid_threads));

  const char *dir = "/tmp/az_fast_bad_best";
  mkdir(dir, 0755);
  const std::string student_path = std::string(dir) + "/student.net";
  const std::string best_path = std::string(dir) + "/best.net";
  deeplearning::PolicyValueResNet student, mismatched_best;
  CHECK(student.Init(TinyNetConfig(3)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  auto mismatched = TinyNetConfig(4);
  mismatched.trunk_channels_ = 8;
  CHECK(mismatched_best.Init(mismatched) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(student.Save(student_path) == deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(mismatched_best.Save(best_path) ==
        deeplearning::PolicyValueResNet::SUCCESS);

  az::TrainConfig config;
  config.net_ = TinyNetConfig(3);
  config.run_dir_ = dir;
  config.resume_ = false;
  config.buffer_capacity_ = 2;
  config.init_model_ = student_path;
  config.init_best_model_ = best_path;
  az::Trainer trainer;
  CHECK(!trainer.Init(config));

  std::remove(student_path.c_str());
  std::remove(best_path.c_str());
  rmdir(dir);
}

void TestPolicyHeadOnlyTrainerFreezesTrunkValueAndBatchNorm() {
  const char *dir = "/tmp/az_policy_head_only";
  std::system("rm -rf /tmp/az_policy_head_only");
  CHECK(mkdir(dir, 0755) == 0);
  const std::string initial_path = std::string(dir) + "/initial.net";

  deeplearning::PolicyValueResNet initial;
  CHECK(initial.Init(TinyNetConfig(101)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(initial.Save(initial_path) == deeplearning::PolicyValueResNet::SUCCESS);

  std::vector<std::string> names;
  std::vector<std::vector<float>> before_values;
  for (const auto &parameter : initial.TrainableParameters()) {
    names.push_back(parameter.name_);
    before_values.push_back(*parameter.value_);
  }
  std::vector<std::vector<float>> before_running;
  auto append_running = [&](const deeplearning::BatchNorm2D &normalization) {
    before_running.push_back(normalization.running_mean());
    before_running.push_back(normalization.running_variance());
  };
  append_running(initial.stem_norm());
  for (const auto &block : initial.blocks()) {
    append_running(block.norm1());
    append_running(block.norm2());
  }
  append_running(initial.policy_norm());
  append_running(initial.value_norm());

  az::TrainConfig config;
  config.net_ = TinyNetConfig(101);
  config.run_dir_ = dir;
  config.iterations_ = 1;
  config.resume_ = false;
  config.init_model_ = initial_path;
  config.init_best_model_ = initial_path;
  config.policy_head_only_ = true;
  config.selfplay_.worker_num_ = 1;
  config.selfplay_.game_num_ = 1;
  config.selfplay_.mcts_.simulation_num_ = 1;
  config.selfplay_.mcts_.dirichlet_epsilon_ = 0.0f;
  config.selfplay_.temperature_move_cutoff_ = 0;
  config.selfplay_.max_moves_ = 8;
  config.train_steps_ = 1;
  config.batch_size_ = 1;
  config.buffer_capacity_ = 64;
  config.hard_fraction_ = 0.0f;
  config.gate_every_ = 100;
  config.save_buffer_every_ = 1;
  az::Trainer trainer;
  CHECK(trainer.Init(config));
  trainer.Run();

  deeplearning::PolicyValueResNet trained;
  CHECK(trained.Init(TinyNetConfig(101)) ==
        deeplearning::PolicyValueResNet::SUCCESS);
  CHECK(trained.Load(std::string(dir) + "/checkpoint.latest.1.net") ==
        deeplearning::PolicyValueResNet::SUCCESS);
  const auto after_parameters = trained.TrainableParameters();
  CHECK(after_parameters.size() == before_values.size());
  bool policy_changed = false;
  for (std::size_t index = 0; index < after_parameters.size(); ++index) {
    CHECK(after_parameters[index].name_ == names[index]);
    const bool policy = names[index].compare(0, 7, "policy.") == 0;
    if (policy)
      policy_changed = policy_changed ||
                       *after_parameters[index].value_ != before_values[index];
    else
      CHECK(*after_parameters[index].value_ == before_values[index]);
  }
  CHECK(policy_changed);

  std::vector<std::vector<float>> after_running;
  auto append_after = [&](const deeplearning::BatchNorm2D &normalization) {
    after_running.push_back(normalization.running_mean());
    after_running.push_back(normalization.running_variance());
  };
  append_after(trained.stem_norm());
  for (const auto &block : trained.blocks()) {
    append_after(block.norm1());
    append_after(block.norm2());
  }
  append_after(trained.policy_norm());
  append_after(trained.value_norm());
  CHECK(after_running == before_running);

  az::TrainConfig resume = config;
  resume.iterations_ = 2;
  resume.resume_ = true;
  resume.init_model_.clear();
  resume.init_best_model_.clear();
  az::Trainer resumed_trainer;
  CHECK(resumed_trainer.Init(resume));
  resumed_trainer.Run();
  CHECK(access((std::string(dir) + "/checkpoint.latest.2.net").c_str(),
               F_OK) == 0);

  az::TrainConfig wrong_mode = resume;
  wrong_mode.iterations_ = 3;
  wrong_mode.policy_head_only_ = false;
  az::Trainer wrong_trainer;
  CHECK(!wrong_trainer.Init(wrong_mode));
  std::system("rm -rf /tmp/az_policy_head_only");
}

void TestBatchNormFixedStatisticsBackward() {
  deeplearning::BatchNorm2D norm;
  deeplearning::BatchNorm2D::Config config;
  config.channels_ = 1;
  CHECK(norm.Init(config) == deeplearning::BatchNorm2D::SUCCESS);
  CHECK(norm.set_scale({1.2f}) == deeplearning::BatchNorm2D::SUCCESS);
  CHECK(norm.set_bias({-0.3f}) == deeplearning::BatchNorm2D::SUCCESS);
  CHECK(norm.set_running_mean({0.25f}) ==
        deeplearning::BatchNorm2D::SUCCESS);
  CHECK(norm.set_running_variance({1.5f}) ==
        deeplearning::BatchNorm2D::SUCCESS);
  deeplearning::FloatTensor4D input(1, 1, 1, 2);
  input(0, 0, 0, 0) = 0.7f;
  input(0, 0, 0, 1) = -0.2f;
  deeplearning::FloatTensor4D output;
  CHECK(norm.Forward(input, output, true, true) ==
        deeplearning::BatchNorm2D::SUCCESS);
  deeplearning::FloatTensor4D grad_output(1, 1, 1, 2);
  grad_output(0, 0, 0, 0) = 0.7f;
  grad_output(0, 0, 0, 1) = -0.4f;
  deeplearning::FloatTensor4D grad_input;
  CHECK(norm.Backward(grad_output, grad_input) ==
        deeplearning::BatchNorm2D::SUCCESS);
  const float coefficient = 1.2f / std::sqrt(1.5f + config.epsilon_);
  CHECK(std::fabs(grad_input(0, 0, 0, 0) - coefficient * 0.7f) < 1e-6f);
  CHECK(std::fabs(grad_input(0, 0, 0, 1) + coefficient * 0.4f) < 1e-6f);
  CHECK(norm.running_mean() == std::vector<float>({0.25f}));
  CHECK(norm.running_variance() == std::vector<float>({1.5f}));
}

} // namespace

int main() {
  TestHorizontalWin();
  TestVerticalAndDiagonalWin();
  TestIllegalMoves();
  TestDraw();
  TestEncode();
  TestEvalCacheIncludesLastMove();
  TestSymmetryConsistency();
  TestMctsForcedWin();
  TestVisitDistribution();
  TestMctsTreeReuse();
  TestMctsReuseDisabledByDefault();
  TestMctsReuseSafetyAndBounds();
  TestMctsNoiseResetOnReuse();
  TestDirichletNoiseWeight();
  TestReplaySaveReportsWriteFailure();
  TestReplayLoadFailurePreservesBuffer();
  TestTeacherTargetsDoNotControlBehavior();
  TestEvalCacheCountersAreThreadSafe();
  TestDynamicBatchEvaluatorMatchesSingleEvaluation();
  TestDynamicBatchEvaluatorFlushesPartialBatch();
  TestDynamicBatchEvaluatorDrainsOnStop();
  TestPolicyValueResNetExpansionPreservesInference();
  TestExpandedTrainerCheckpointResume();
  TestFreshTrainerLoadsInitialReplay();
  TestSelfPlayUsesDynamicBatchEvaluator();
  TestExpandedStudentSupportsSmallerTeacher();
  TestFastTrainerRejectsInvalidConfigAndMismatchedBest();
  TestPolicyHeadOnlyTrainerFreezesTrunkValueAndBatchNorm();
  TestBatchNormFixedStatisticsBackward();

  std::printf("%d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
