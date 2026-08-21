#include "train/evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>

namespace az {

namespace {

bool CopyBatchNormState(deeplearning::BatchNorm2D &dst,
                        const deeplearning::BatchNorm2D &src) {
  if (dst.config().channels_ != src.config().channels_) {
    return false;
  }
  return dst.set_running_mean(src.running_mean()) ==
             deeplearning::BatchNorm2D::SUCCESS &&
         dst.set_running_variance(src.running_variance()) ==
             deeplearning::BatchNorm2D::SUCCESS;
}

} // namespace

// Copies trainable parameters AND batch-norm running statistics between two
// networks with identical configs. Returns false on any shape mismatch.
bool AssignWeights(deeplearning::PolicyValueResNet &dst,
                   deeplearning::PolicyValueResNet &src) {
  auto dst_params = dst.TrainableParameters();
  auto src_params = src.TrainableParameters();
  if (dst_params.size() != src_params.size()) {
    return false;
  }
  for (std::size_t i = 0; i < dst_params.size(); ++i) {
    if (dst_params[i].value_->size() != src_params[i].value_->size()) {
      return false;
    }
    *dst_params[i].value_ = *src_params[i].value_;
  }
  // Batch-norm running statistics are buffers, not trainable parameters:
  // copy them explicitly or inference would normalize with mean=0/var=1.
  if (!CopyBatchNormState(dst.stem_norm(), src.stem_norm()) ||
      !CopyBatchNormState(dst.policy_norm(), src.policy_norm()) ||
      !CopyBatchNormState(dst.value_norm(), src.value_norm())) {
    return false;
  }
  auto &dst_blocks = dst.blocks();
  auto &src_blocks = src.blocks();
  if (dst_blocks.size() != src_blocks.size()) {
    return false;
  }
  for (std::size_t i = 0; i < dst_blocks.size(); ++i) {
    if (!CopyBatchNormState(dst_blocks[i].norm1(), src_blocks[i].norm1()) ||
        !CopyBatchNormState(dst_blocks[i].norm2(), src_blocks[i].norm2())) {
      return false;
    }
  }
  return true;
}

void Evaluator::Predict(const Gomoku &game, float *policy, float &value) {
  game.Encode(input_);
  net_.Forward(input_, output_, /*training=*/false);

  value = output_.values_[0];

  // Legal-move masked softmax.
  float max_logit = -1e30f;
  for (int action = 0; action < Gomoku::kActionNum; ++action) {
    if (game.IsLegal(action)) {
      max_logit = std::max(max_logit, output_.policy_logits_[action]);
    }
  }
  float sum = 0.0f;
  for (int action = 0; action < Gomoku::kActionNum; ++action) {
    if (game.IsLegal(action)) {
      const float p = std::exp(output_.policy_logits_[action] - max_logit);
      policy[action] = p;
      sum += p;
    } else {
      policy[action] = 0.0f;
    }
  }
  if (sum > 0.0f) {
    const float inverse = 1.0f / sum;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      policy[action] *= inverse;
    }
  } else {
    // Defensive fallback: uniform over legal moves.
    int legal = 0;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (game.IsLegal(action)) ++legal;
    }
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (game.IsLegal(action)) policy[action] = 1.0f / legal;
    }
  }
}

bool DynamicBatchEvaluator::Init(
    deeplearning::PolicyValueResNet &master, const Config &config) {
  if (config.max_batch_size_ <= 0 || config.max_wait_us_ < 0 ||
      config.inference_thread_num_ <= 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (started_) return false;
  config_ = config;
  auto net_config = master.config();
  net_config.thread_num_ = config.inference_thread_num_;
  if (net_.Init(net_config) != deeplearning::PolicyValueResNet::SUCCESS ||
      !AssignWeights(net_, master)) {
    return false;
  }
  // Build lazy packed convolution weights before callers can enqueue work.
  Gomoku warmup_game;
  warmup_game.Encode(input_);
  if (net_.Forward(input_, output_, false) !=
      deeplearning::PolicyValueResNet::SUCCESS) {
    return false;
  }
  started_ = true;
  stopping_ = false;
  worker_ = std::thread(&DynamicBatchEvaluator::Run, this);
  return true;
}

void DynamicBatchEvaluator::Predict(const Gomoku &game, float *policy,
                                    float &value) {
  auto request = std::make_shared<Request>();
  request->game_ = game;
  request->enqueue_time_ = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (!started_ || stopping_) {
      SetFallback(*request);
      std::copy(request->policy_.begin(), request->policy_.end(), policy);
      value = request->value_;
      return;
    }
    queue_.push_back(request);
    requests_.fetch_add(1, std::memory_order_relaxed);
  }
  queue_condition_.notify_one();

  std::unique_lock<std::mutex> lock(request->mutex_);
  request->condition_.wait(lock, [&]() { return request->done_; });
  std::copy(request->policy_.begin(), request->policy_.end(), policy);
  value = request->value_;
}

void DynamicBatchEvaluator::Stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (!started_) return;
    stopping_ = true;
  }
  queue_condition_.notify_all();
  if (worker_.joinable()) worker_.join();
  std::lock_guard<std::mutex> lock(queue_mutex_);
  started_ = false;
}

DynamicBatchEvaluator::Stats DynamicBatchEvaluator::GetStats() const {
  Stats stats;
  stats.requests_ = requests_.load(std::memory_order_relaxed);
  stats.forward_calls_ = forward_calls_.load(std::memory_order_relaxed);
  stats.total_batch_size_ = total_batch_size_.load(std::memory_order_relaxed);
  stats.total_queue_wait_us_ =
      total_queue_wait_us_.load(std::memory_order_relaxed);
  stats.max_batch_size_ = max_batch_size_.load(std::memory_order_relaxed);
  return stats;
}

void DynamicBatchEvaluator::Run() {
  while (true) {
    std::vector<std::shared_ptr<Request>> batch;
    batch.reserve(static_cast<std::size_t>(config_.max_batch_size_));
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_condition_.wait(lock,
                          [&]() { return stopping_ || !queue_.empty(); });
    if (queue_.empty() && stopping_) break;

    batch.push_back(queue_.front());
    queue_.pop_front();
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::microseconds(config_.max_wait_us_);
    while (static_cast<int>(batch.size()) < config_.max_batch_size_) {
      if (!queue_.empty()) {
        batch.push_back(queue_.front());
        queue_.pop_front();
        continue;
      }
      if (stopping_ || config_.max_wait_us_ == 0) break;
      if (!queue_condition_.wait_until(lock, deadline,
                                        [&]() { return stopping_ ||
                                                        !queue_.empty(); })) {
        break;
      }
      if (queue_.empty() && stopping_) break;
    }
    lock.unlock();
    EvaluateBatch(batch);
  }
}

void DynamicBatchEvaluator::EvaluateBatch(
    const std::vector<std::shared_ptr<Request>> &batch) {
  const int batch_size = static_cast<int>(batch.size());
  input_.Resize(batch_size, Gomoku::kPlaneNum, Gomoku::kBoardSize,
                Gomoku::kBoardSize);
  const int plane_size = Gomoku::kPlaneNum * Gomoku::kCellNum;
  const auto now = std::chrono::steady_clock::now();
  for (int index = 0; index < batch_size; ++index) {
    batch[index]->game_.EncodeInto(input_.data() + index * plane_size);
    const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - batch[index]->enqueue_time_).count();
    if (wait_us > 0) {
      total_queue_wait_us_.fetch_add(static_cast<std::uint64_t>(wait_us),
                                     std::memory_order_relaxed);
    }
  }

  const bool success =
      net_.Forward(input_, output_, false) ==
          deeplearning::PolicyValueResNet::SUCCESS &&
      output_.batch_ == batch_size &&
      output_.policy_logits_.size() ==
          static_cast<std::size_t>(batch_size * Gomoku::kActionNum) &&
      output_.values_.size() == static_cast<std::size_t>(batch_size);

  forward_calls_.fetch_add(1, std::memory_order_relaxed);
  total_batch_size_.fetch_add(static_cast<std::uint64_t>(batch_size),
                              std::memory_order_relaxed);
  int current_max = max_batch_size_.load(std::memory_order_relaxed);
  while (batch_size > current_max &&
         !max_batch_size_.compare_exchange_weak(
             current_max, batch_size, std::memory_order_relaxed)) {}

  for (int index = 0; index < batch_size; ++index) {
    Request &request = *batch[index];
    if (!success) {
      SetFallback(request);
    } else {
      request.value_ = output_.values_[index];
      const int offset = index * Gomoku::kActionNum;
      float max_logit = -1e30f;
      for (int action = 0; action < Gomoku::kActionNum; ++action) {
        if (request.game_.IsLegal(action)) {
          max_logit = std::max(max_logit,
              output_.policy_logits_[offset + action]);
        }
      }
      float sum = 0.0f;
      for (int action = 0; action < Gomoku::kActionNum; ++action) {
        if (request.game_.IsLegal(action)) {
          const float probability = std::exp(
              output_.policy_logits_[offset + action] - max_logit);
          request.policy_[action] = probability;
          sum += probability;
        } else {
          request.policy_[action] = 0.0f;
        }
      }
      if (sum > 0.0f) {
        const float inverse = 1.0f / sum;
        for (float &probability : request.policy_) probability *= inverse;
      } else {
        SetFallback(request);
      }
    }
    {
      std::lock_guard<std::mutex> lock(request.mutex_);
      request.done_ = true;
    }
    request.condition_.notify_one();
  }
}

void DynamicBatchEvaluator::SetFallback(Request &request) {
  int legal = 0;
  for (int action = 0; action < Gomoku::kActionNum; ++action) {
    if (request.game_.IsLegal(action)) ++legal;
  }
  request.policy_.fill(0.0f);
  if (legal > 0) {
    const float probability = 1.0f / legal;
    for (int action = 0; action < Gomoku::kActionNum; ++action) {
      if (request.game_.IsLegal(action)) request.policy_[action] = probability;
    }
  }
  request.value_ = 0.0f;
}

// --- EvalCache ---

std::string EvalCache::EncodeKey(const Gomoku &game) {
  std::string key(Gomoku::kCellNum + 2, '\0');
  const int player = game.current_player();
  const auto &board = game.board();
  key[0] = static_cast<char>(player);
  // +1 keeps -1 (no last move) distinguishable from action 0.
  key[1] = static_cast<char>(game.last_action() + 1);
  for (int cell = 0; cell < Gomoku::kCellNum; ++cell) {
    // 0 empty, 1 own stone, 2 opponent stone (state is player-relative)
    key[cell + 2] = board[cell] == 0
                        ? '\0'
                        : static_cast<char>(board[cell] == player ? 1 : 2);
  }
  return key;
}

bool EvalCache::Lookup(const Gomoku &game, Entry &entry) {
  const std::string key = EncodeKey(game);
  const int shard = static_cast<int>(std::hash<std::string>{}(key) % kShardNum);
  std::lock_guard<std::mutex> lock(mutex_[shard]);
  ++lookups_;
  auto it = shards_[shard].find(key);
  if (it == shards_[shard].end()) {
    return false;
  }
  ++hits_;
  entry = it->second;
  return true;
}

void EvalCache::Store(const Gomoku &game, const float *policy, float value) {
  const std::string key = EncodeKey(game);
  const int shard = static_cast<int>(std::hash<std::string>{}(key) % kShardNum);
  Entry entry;
  std::copy(policy, policy + Gomoku::kActionNum, entry.policy.begin());
  entry.value = value;
  std::lock_guard<std::mutex> lock(mutex_[shard]);
  shards_[shard].emplace(std::move(key), std::move(entry));
}

void EvalCache::Clear() {
  for (int shard = 0; shard < kShardNum; ++shard) {
    std::lock_guard<std::mutex> lock(mutex_[shard]);
    shards_[shard].clear();
  }
  lookups_.store(0);
  hits_.store(0);
}

std::size_t EvalCache::Size() const {
  std::size_t total = 0;
  for (int shard = 0; shard < kShardNum; ++shard) {
    std::lock_guard<std::mutex> lock(mutex_[shard]);
    total += shards_[shard].size();
  }
  return total;
}

} // namespace az
