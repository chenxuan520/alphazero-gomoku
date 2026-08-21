#pragma once

#include "cnn/policy_value_resnet.h"
#include "game/gomoku.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace az {

// Interface MCTS uses for position evaluation. The neural-net Evaluator and
// test doubles both implement this.
class INetEvaluator {
public:
  virtual ~INetEvaluator() = default;
  // policy output: probabilities over all kActionNum actions (0 on illegal
  // moves, sums to 1 over legal ones). value: expected game outcome in
  // [-1, 1] from the perspective of the player to move.
  virtual void Predict(const Gomoku &game, float *policy, float &value) = 0;
};

// Copies all trainable parameters AND batch-norm running statistics (same
// order, identical configs required). Returns false on any shape mismatch.
bool AssignWeights(deeplearning::PolicyValueResNet &dst,
                   deeplearning::PolicyValueResNet &src);

// Neural-net evaluator. NOT thread-safe: give every worker its own copy.
class Evaluator : public INetEvaluator {
public:
  bool Init(const deeplearning::PolicyValueResNet::Config &config) {
    return net_.Init(config) == deeplearning::PolicyValueResNet::SUCCESS;
  }
  void Predict(const Gomoku &game, float *policy, float &value) override;

  deeplearning::PolicyValueResNet &net() { return net_; }

private:
  deeplearning::PolicyValueResNet net_;
  deeplearning::FloatTensor4D input_;
  deeplearning::PolicyValueResNet::Output output_;
};

// One-owner dynamic batching service. Callers keep the synchronous evaluator
// API, but cache misses are queued and packed into one network forward. The
// network is touched only by worker_, because PolicyValueResNet inference is
// not safe for concurrent direct calls.
class DynamicBatchEvaluator : public INetEvaluator {
public:
  struct Config {
    int max_batch_size_ = 32;
    int max_wait_us_ = 200;
    int inference_thread_num_ = 1;
  };

  struct Stats {
    std::uint64_t requests_ = 0;
    std::uint64_t forward_calls_ = 0;
    std::uint64_t total_batch_size_ = 0;
    std::uint64_t total_queue_wait_us_ = 0;
    int max_batch_size_ = 0;
  };

  DynamicBatchEvaluator() = default;
  ~DynamicBatchEvaluator() override { Stop(); }
  DynamicBatchEvaluator(const DynamicBatchEvaluator &) = delete;
  DynamicBatchEvaluator &operator=(const DynamicBatchEvaluator &) = delete;

  bool Init(deeplearning::PolicyValueResNet &master, const Config &config);
  void Predict(const Gomoku &game, float *policy, float &value) override;
  void Stop();
  Stats GetStats() const;

private:
  struct Request {
    Gomoku game_;
    std::array<float, Gomoku::kActionNum> policy_{};
    float value_ = 0.0f;
    bool done_ = false;
    std::chrono::steady_clock::time_point enqueue_time_;
    std::mutex mutex_;
    std::condition_variable condition_;
  };

  void Run();
  void EvaluateBatch(const std::vector<std::shared_ptr<Request>> &batch);
  static void SetFallback(Request &request);

  Config config_;
  deeplearning::PolicyValueResNet net_;
  deeplearning::FloatTensor4D input_;
  deeplearning::PolicyValueResNet::Output output_;
  std::thread worker_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_condition_;
  std::deque<std::shared_ptr<Request>> queue_;
  bool started_ = false;
  bool stopping_ = false;
  std::atomic<std::uint64_t> requests_{0};
  std::atomic<std::uint64_t> forward_calls_{0};
  std::atomic<std::uint64_t> total_batch_size_{0};
  std::atomic<std::uint64_t> total_queue_wait_us_{0};
  std::atomic<int> max_batch_size_{0};
};

// Thread-safe evaluation cache: exact board-position key (this player's /
// opponent's stone layout as seen by the player to move) -> net output.
// Valid only while net weights are fixed; Clear() after every weight update.
// Pure memory-for-CPU: repeated positions across concurrent games skip the
// expensive forward pass.
class EvalCache {
public:
  struct Entry {
    std::array<float, Gomoku::kActionNum> policy;
    float value;
  };

  bool Lookup(const Gomoku &game, Entry &entry);
  void Store(const Gomoku &game, const float *policy, float value);
  void Clear();
  std::size_t Size() const;

  // stats for logs
  std::size_t Lookups() const { return lookups_.load(); }
  std::size_t Hits() const { return hits_.load(); }

private:
  static constexpr int kShardNum = 64;
  static std::string EncodeKey(const Gomoku &game);

  mutable std::mutex mutex_[kShardNum];
  std::unordered_map<std::string, Entry> shards_[kShardNum];
  std::atomic<std::size_t> lookups_{0};
  std::atomic<std::size_t> hits_{0};
};

// Decorator adding cache behavior on top of another evaluator.
class CachedEvaluator : public INetEvaluator {
public:
  CachedEvaluator(INetEvaluator *inner, EvalCache *cache)
      : inner_(inner), cache_(cache) {}
  void Predict(const Gomoku &game, float *policy, float &value) override {
    if (cache_ != nullptr) {
      EvalCache::Entry entry;
      if (cache_->Lookup(game, entry)) {
        std::copy(entry.policy.begin(), entry.policy.end(), policy);
        value = entry.value;
        return;
      }
    }
    inner_->Predict(game, policy, value);
    if (cache_ != nullptr) {
      cache_->Store(game, policy, value);
    }
  }

private:
  INetEvaluator *inner_;
  EvalCache *cache_;
};

} // namespace az
