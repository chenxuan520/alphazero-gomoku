#include "train/self_play.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

namespace az {

namespace {

// Reconstructs a game state from a buffer sample's planes (mover-relative:
// own = mover stones, side plane = 1 if mover is black). Returns false on
// inconsistency.
bool GameFromSample(const Sample &sample, Gomoku &game) {
  game.Reset();
  int mover_stones = 0, other_stones = 0;
  int last_action = -1;
  const int side_is_black = sample.planes[3 * 225] > 0.5f ? 1 : 0;
  const int8_t mover = side_is_black ? Gomoku::kBlack : Gomoku::kWhite;
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    if (sample.planes[cell] > 0.5f) {
      game.MutableBoard()[cell] = static_cast<int8_t>(mover);
      ++mover_stones;
    } else if (sample.planes[Gomoku::kCellNum + cell] > 0.5f) {
      game.MutableBoard()[cell] = static_cast<int8_t>(-mover);
      ++other_stones;
    }
    if (sample.planes[2 * Gomoku::kCellNum + cell] > 0.5f) {
      if (last_action >= 0) return false;
      last_action = cell;
    }
  }
  // In a legal position, the mover has same or one fewer stone than opponent
  // when side==black... simplest sanity: total stones even if black to move.
  const int total = mover_stones + other_stones;
  if (side_is_black && total % 2 != 0) return false;
  if (!side_is_black && total % 2 != 1) return false;
  game.SetState(mover, total, /*result=*/0, last_action);
  return true;
}

// Plays a single self-play game; appends samples (with final z values) to the
// buffer. Returns the move count.
int PlaySelfPlayGame(INetEvaluator &evaluator, EvalCache *cache,
                       INetEvaluator *teacher_evaluator, EvalCache *teacher_cache,
                      const SelfPlayConfig &config, ReplayBuffer &buffer,
                      const ReplayBuffer *seed_source, std::mt19937 &rng,
                      std::mt19937 &teacher_rng, int &teacher_target_count,
                      int &sample_count) {
  Mcts mcts;
  CachedEvaluator cached(&evaluator, cache);
  INetEvaluator *used = cache != nullptr
                            ? static_cast<INetEvaluator *>(&cached)
                            : static_cast<INetEvaluator *>(&evaluator);
  CachedEvaluator cached_teacher(teacher_evaluator, teacher_cache);
  INetEvaluator *teacher_used = teacher_evaluator == nullptr
      ? nullptr
      : (teacher_cache != nullptr
             ? static_cast<INetEvaluator *>(&cached_teacher)
             : static_cast<INetEvaluator *>(teacher_evaluator));
  std::unique_ptr<Mcts> teacher_mcts;
  if (teacher_used != nullptr) teacher_mcts.reset(new Mcts());

  Gomoku game;
  // Curriculum seeding: optionally start from a hard defensive position.
  if (config.hard_seed_indices_ != nullptr && seed_source != nullptr &&
      !config.hard_seed_indices_->empty()) {
    std::uniform_real_distribution<float> roll(0.0f, 1.0f);
    if (roll(rng) < config.seed_from_hard_prob_) {
      std::uniform_int_distribution<int> pick(
          0, static_cast<int>(config.hard_seed_indices_->size()) - 1);
      const int idx = (*config.hard_seed_indices_)[pick(rng)];
      if (idx >= 0 &&
          static_cast<std::size_t>(idx) < seed_source->Size()) {
        Gomoku seeded;
        if (GameFromSample(seed_source->At(idx), seeded)) {
          game = seeded;
        }
      }
    }
  }
  std::vector<Sample> history;
  std::vector<int> history_player;
  std::vector<int> visit_action, visit_count;
  std::vector<int> teacher_action, teacher_count;
  std::array<float, Gomoku::kActionNum> behavior_policy;
  std::uniform_real_distribution<float> teacher_roll(0.0f, 1.0f);
  Sample sample;

  while (!game.IsTerminal() && game.move_count() < config.max_moves_) {
    mcts.Search(game, config.mcts_, *used, rng, visit_action, visit_count);

    Mcts::VisitDistribution(visit_action, visit_count, behavior_policy.data());
    game.EncodeInto(sample.planes.data());
    const bool use_teacher = teacher_used != nullptr &&
        config.teacher_target_prob_ > 0.0f &&
        teacher_roll(teacher_rng) < config.teacher_target_prob_;
    if (use_teacher) {
      MctsConfig teacher_config = config.mcts_;
      teacher_config.simulation_num_ = config.teacher_simulations_;
      teacher_config.dirichlet_epsilon_ = 0.0f;
      teacher_config.reuse_tree_ = false;
      teacher_mcts->Search(game, teacher_config, *teacher_used, teacher_rng,
                           teacher_action, teacher_count);
      Mcts::VisitDistribution(teacher_action, teacher_count,
                              sample.policy.data());
      ++teacher_target_count;
    } else {
      sample.policy = behavior_policy;
    }
    history.push_back(sample);
    history_player.push_back(game.current_player());

    // pick the action: sample during the opening, argmax later
    int action = -1;
    if (game.move_count() < config.temperature_move_cutoff_) {
      std::discrete_distribution<int> dist(behavior_policy.begin(),
                                           behavior_policy.end());
      action = dist(rng);
    } else {
      action = static_cast<int>(
          std::max_element(behavior_policy.begin(), behavior_policy.end()) -
          behavior_policy.begin());
    }
    // dirichlet noise can spread counts onto anything legal; guard anyway
    if (action < 0 || !game.IsLegal(action)) {
      std::vector<int> legal;
      for (int a = 0; a < Gomoku::kActionNum; ++a)
        if (game.IsLegal(a)) legal.push_back(a);
      std::uniform_int_distribution<int> dist2(0,
                                               (int)legal.size() - 1);
      action = legal[dist2(rng)];
    }
    game.Apply(action);
    if (config.mcts_.reuse_tree_) mcts.AdvanceRoot(action);
  }

  const int result = game.IsTerminal() ? game.Result() : 2; // cap -> draw
  const int winner = (result == 2) ? 0 : result;
  for (std::size_t i = 0; i < history.size(); ++i) {
    history[i].value =
        winner == 0 ? 0.0f : (history_player[i] == winner ? 1.0f : -1.0f);
    buffer.Push(history[i]);
  }
  sample_count = static_cast<int>(history.size());
  return game.move_count();
}

} // namespace

SelfPlayStats RunSelfPlay(deeplearning::PolicyValueResNet &master,
                          const SelfPlayConfig &config,
                          ReplayBuffer &buffer,
                          deeplearning::PolicyValueResNet *teacher) {
  EvalCache cache;
  EvalCache *cache_ptr = config.use_cache_ ? &cache : nullptr;
  EvalCache teacher_cache;
  EvalCache *teacher_cache_ptr =
      config.use_cache_ && teacher != nullptr ? &teacher_cache : nullptr;

  DynamicBatchEvaluator student_service;
  DynamicBatchEvaluator teacher_service;
  DynamicBatchEvaluator::Config service_config;
  service_config.max_batch_size_ = config.inference_batch_size_;
  service_config.max_wait_us_ = config.inference_wait_us_;
  service_config.inference_thread_num_ = config.inference_thread_num_;
  if (config.use_batch_inference_ &&
      !student_service.Init(master, service_config)) {
    std::fprintf(stderr, "[selfplay] student batch evaluator init failed\n");
    return SelfPlayStats();
  }
  if (config.use_batch_inference_ && teacher != nullptr &&
      !teacher_service.Init(*teacher, service_config)) {
    std::fprintf(stderr, "[selfplay] teacher batch evaluator init failed\n");
    student_service.Stop();
    return SelfPlayStats();
  }

  // Snapshot worker-local evaluators serially. AssignWeights touches the
  // source network's lazy packed-weight flags, so concurrent copies formally
  // race even though parameter payloads are read-only.
  std::vector<std::unique_ptr<Evaluator>> local_evaluators;
  std::vector<std::unique_ptr<Evaluator>> local_teachers;
  if (!config.use_batch_inference_) {
    local_evaluators.resize(config.worker_num_);
    local_teachers.resize(config.worker_num_);
    auto net_config = master.config();
    net_config.thread_num_ = 1;
    auto teacher_config = teacher != nullptr ? teacher->config() : net_config;
    teacher_config.thread_num_ = 1;
    for (int worker = 0; worker < config.worker_num_; ++worker) {
      local_evaluators[worker].reset(new Evaluator());
      if (!local_evaluators[worker]->Init(net_config) ||
          !AssignWeights(local_evaluators[worker]->net(), master)) {
        std::fprintf(stderr, "[selfplay] worker %d net init failed\n", worker);
        return SelfPlayStats();
      }
      if (teacher != nullptr) {
        local_teachers[worker].reset(new Evaluator());
        if (!local_teachers[worker]->Init(teacher_config) ||
            !AssignWeights(local_teachers[worker]->net(), *teacher)) {
          std::fprintf(stderr, "[selfplay] worker %d teacher init failed\n",
                       worker);
          return SelfPlayStats();
        }
      }
    }
  }

  std::atomic<int> next_game{0};
  std::atomic<int> finished{0};
  SelfPlayStats stats;
  std::mutex stats_mutex;

  auto worker = [&](int worker_index) {
    INetEvaluator *evaluator_ptr = nullptr;
    INetEvaluator *teacher_ptr = nullptr;
    if (config.use_batch_inference_) {
      evaluator_ptr = &student_service;
      if (teacher != nullptr) teacher_ptr = &teacher_service;
    } else {
      evaluator_ptr = local_evaluators[worker_index].get();
      if (teacher != nullptr) teacher_ptr = local_teachers[worker_index].get();
    }
    std::mt19937 rng(
        static_cast<unsigned>(config.seed_ * 1000003u + worker_index));
    std::mt19937 teacher_rng(static_cast<unsigned>(
        config.seed_ * 2000003u + worker_index + 0x5a17u));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= config.game_num_) {
        break;
      }
      int teacher_targets = 0;
      int generated_samples = 0;
      const int moves = PlaySelfPlayGame(
          *evaluator_ptr, cache_ptr, teacher_ptr, teacher_cache_ptr, config,
          buffer, &buffer, rng, teacher_rng, teacher_targets,
          generated_samples);
      std::lock_guard<std::mutex> lock(stats_mutex);
      ++stats.games;
      stats.moves_total += moves;
      stats.samples += static_cast<std::size_t>(generated_samples);
      stats.teacher_policy_targets +=
          static_cast<std::size_t>(teacher_targets);
      stats.student_policy_targets +=
          static_cast<std::size_t>(generated_samples - teacher_targets);
      const int done = finished.fetch_add(1) + 1;
      if (done % 5 == 0 || done == config.game_num_) {
        std::fprintf(stderr, "\r[selfplay] games %d/%d", done,
                     config.game_num_);
        std::fflush(stderr);
      }
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < config.worker_num_; ++i) {
    threads.emplace_back(worker, i);
  }
  for (auto &t : threads) t.join();
  std::fprintf(stderr, "\n");

  if (config.use_batch_inference_) {
    student_service.Stop();
    teacher_service.Stop();
    const auto batch_stats = student_service.GetStats();
    stats.inference_forward_calls = batch_stats.forward_calls_;
    stats.inference_max_batch = batch_stats.max_batch_size_;
    if (batch_stats.forward_calls_ > 0) {
      stats.inference_average_batch =
          static_cast<double>(batch_stats.total_batch_size_) /
          batch_stats.forward_calls_;
    }
    if (batch_stats.requests_ > 0) {
      stats.inference_average_queue_wait_us =
          static_cast<double>(batch_stats.total_queue_wait_us_) /
          batch_stats.requests_;
    }
  }

  if (cache_ptr != nullptr) {
    stats.eval_cache_size = static_cast<double>(cache.Size());
    stats.eval_cache_hit_rate =
        cache.Lookups() > 0
            ? static_cast<double>(cache.Hits()) / cache.Lookups()
            : 0.0;
  }
  return stats;
}

int PlayMatch(INetEvaluator &black_evaluator, INetEvaluator &white_evaluator,
              const MctsConfig &mcts_config, int temperature_move_cutoff,
              int max_moves, std::mt19937 &rng,
              const MctsConfig *mcts_config_white) {
  MctsConfig match_config = mcts_config;
  // Preserve historical match/gate behavior unless a caller explicitly opts
  // into the standards-compliant noisy evaluation mode.
  if (!match_config.normalized_dirichlet_)
    match_config.dirichlet_epsilon_ = 0.0f;
  MctsConfig white_config =
      mcts_config_white ? *mcts_config_white : match_config;
  if (!white_config.normalized_dirichlet_)
    white_config.dirichlet_epsilon_ = 0.0f;
  Mcts mcts_black, mcts_white;
  Gomoku game;
  std::vector<int> visit_action, visit_count;
  std::array<float, Gomoku::kActionNum> pi;

  while (!game.IsTerminal() && game.move_count() < max_moves) {
    INetEvaluator &evaluator =
        game.current_player() == Gomoku::kBlack ? black_evaluator
                                                : white_evaluator;
    Mcts &mcts =
        game.current_player() == Gomoku::kBlack ? mcts_black : mcts_white;
    const MctsConfig &side_config = game.current_player() == Gomoku::kBlack
                                        ? match_config
                                        : white_config;
    mcts.Search(game, side_config, evaluator, rng, visit_action, visit_count);
    Mcts::VisitDistribution(visit_action, visit_count, pi.data());
    int action;
    if (game.move_count() < temperature_move_cutoff) {
      std::discrete_distribution<int> dist(pi.begin(), pi.end());
      action = dist(rng);
    } else {
      action = static_cast<int>(std::max_element(pi.begin(), pi.end()) -
                                pi.begin());
    }
    if (!game.IsLegal(action)) {
      // fall back to the highest-prior legal move
      float best = -1.0f;
      for (int a = 0; a < Gomoku::kActionNum; ++a)
        if (game.IsLegal(a) && pi[a] > best) {
          best = pi[a];
          action = a;
        }
    }
    game.Apply(action);
    if (match_config.reuse_tree_) {
      mcts_black.AdvanceRoot(action);
      mcts_white.AdvanceRoot(action);
    }
  }
  return game.IsTerminal() ? game.Result() : 2;
}

void RandomEvaluator::Predict(const Gomoku &game, float *policy,
                              float &value) {
  int legal = 0;
  for (int a = 0; a < Gomoku::kActionNum; ++a)
    if (game.IsLegal(a)) ++legal;
  for (int a = 0; a < Gomoku::kActionNum; ++a)
    policy[a] = game.IsLegal(a) ? 1.0f / legal : 0.0f;
  value = 0.0f;
}

DuelStats RunDuel(deeplearning::PolicyValueResNet &a,
                  deeplearning::PolicyValueResNet &b,
                  const MctsConfig &mcts_config, int game_num, int worker_num,
                  int temperature_move_cutoff, int max_moves, int seed,
                  int sims_b) {
  DuelStats stats;
  MctsConfig mcts_config_b = mcts_config;
  if (sims_b > 0) mcts_config_b.simulation_num_ = sims_b;
  std::mutex mutex;
  std::atomic<int> next_game{0};
  std::vector<std::unique_ptr<Evaluator>> evaluators_a(worker_num);
  std::vector<std::unique_ptr<Evaluator>> evaluators_b(worker_num);
  auto config_a = a.config();
  auto config_b = b.config();
  config_a.thread_num_ = 1;
  config_b.thread_num_ = 1;
  for (int worker = 0; worker < worker_num; ++worker) {
    evaluators_a[worker].reset(new Evaluator());
    evaluators_b[worker].reset(new Evaluator());
    if (!evaluators_a[worker]->Init(config_a) ||
        !evaluators_b[worker]->Init(config_b) ||
        !AssignWeights(evaluators_a[worker]->net(), a) ||
        !AssignWeights(evaluators_b[worker]->net(), b)) {
      std::fprintf(stderr, "[duel] evaluator snapshot failed\n");
      return stats;
    }
  }

  auto worker = [&](int worker_index) {
    Evaluator &eval_a = *evaluators_a[worker_index];
    Evaluator &eval_b = *evaluators_b[worker_index];
    std::mt19937 worker_rng(
        static_cast<unsigned>(seed * 1000003u + worker_index));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= game_num) break;
      std::mt19937 game_rng;
      std::mt19937 *rng = &worker_rng;
      if (mcts_config.deterministic_game_seeds_) {
        game_rng.seed(static_cast<unsigned>(seed * 1000003u + game_index));
        rng = &game_rng;
      }
      // alternate colors: even games A is black
      INetEvaluator *black = &eval_a, *white = &eval_b;
      bool a_is_black = true;
      if (game_index % 2 == 1) {
        std::swap(black, white);
        a_is_black = false;
      }
      const MctsConfig &black_config =
          a_is_black ? mcts_config : mcts_config_b;
      const MctsConfig &white_config =
          a_is_black ? mcts_config_b : mcts_config;
      const int result = PlayMatch(*black, *white, black_config,
                                   temperature_move_cutoff, max_moves,
                                   *rng, &white_config);
      std::lock_guard<std::mutex> lock(mutex);
      if (result == 2) ++stats.draws;
      else if ((result == Gomoku::kBlack) == a_is_black) {
        ++stats.a_wins;
        if (a_is_black) ++stats.a_black_wins;
        else ++stats.a_white_wins;
      } else ++stats.b_wins;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < worker_num; ++i) threads.emplace_back(worker, i);
  for (auto &t : threads) t.join();
  return stats;
}

DuelStats RunVsRandom(deeplearning::PolicyValueResNet &net,
                      const MctsConfig &mcts_config, int game_num,
                      int worker_num, int max_moves, int seed) {
  DuelStats stats;
  std::mutex mutex;
  std::atomic<int> next_game{0};
  MctsConfig quiet = mcts_config;
  quiet.dirichlet_epsilon_ = 0.0f;
  std::vector<std::unique_ptr<Evaluator>> evaluators(worker_num);
  auto net_config = net.config();
  net_config.thread_num_ = 1;
  for (int worker = 0; worker < worker_num; ++worker) {
    evaluators[worker].reset(new Evaluator());
    if (!evaluators[worker]->Init(net_config) ||
        !AssignWeights(evaluators[worker]->net(), net)) {
      std::fprintf(stderr, "[vs-random] evaluator snapshot failed\n");
      return stats;
    }
  }

  auto worker = [&](int worker_index) {
    Evaluator &eval = *evaluators[worker_index];
    Mcts mcts;
    std::vector<int> visit_action, visit_count;
    std::array<float, Gomoku::kActionNum> pi;
    std::mt19937 rng(static_cast<unsigned>(seed * 7919u + worker_index));
    while (true) {
      const int game_index = next_game.fetch_add(1);
      if (game_index >= game_num) break;
      const bool model_black = (game_index % 2 == 0);
      Gomoku game;
      while (!game.IsTerminal() && game.move_count() < max_moves) {
        int action;
        if ((game.current_player() == Gomoku::kBlack) == model_black) {
          mcts.Search(game, quiet, eval, rng, visit_action, visit_count);
          Mcts::VisitDistribution(visit_action, visit_count, pi.data());
          action = static_cast<int>(
              std::max_element(pi.begin(), pi.end()) - pi.begin());
          if (!game.IsLegal(action)) {
            for (action = 0;
                 action < Gomoku::kActionNum && !game.IsLegal(action); ++action) {
            }
          }
        } else {
          // 真随机: 全盘合法手均匀抽
          std::vector<int> legal;
          for (int a = 0; a < Gomoku::kActionNum; ++a)
            if (game.IsLegal(a)) legal.push_back(a);
          std::uniform_int_distribution<int> dist(0,
                                                  static_cast<int>(legal.size()) - 1);
          action = legal[dist(rng)];
        }
        game.Apply(action);
        if (mcts_config.reuse_tree_) mcts.AdvanceRoot(action);
      }
      const int result = game.IsTerminal() ? game.Result() : 2;
      std::lock_guard<std::mutex> lock(mutex);
      if (result == 2) ++stats.draws;
      else if ((result == Gomoku::kBlack) == model_black) ++stats.a_wins;
      else ++stats.b_wins;
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < worker_num; ++i) threads.emplace_back(worker, i);
  for (auto &t : threads) t.join();
  return stats;
}

} // namespace az
