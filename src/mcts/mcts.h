#pragma once

#include "game/gomoku.h"
#include "game/vcf.h"
#include "train/evaluator.h"

#include <random>
#include <vector>

namespace az {

struct MctsConfig {
  int simulation_num_ = 100;      // tree size per move
  float c_puct_ = 1.5f;           // exploration constant
  float dirichlet_alpha_ = 0.3f;  // root noise shape
  float dirichlet_epsilon_ = 0.25f; // root noise weight (0 disables)
  // Keep the mature training run bit-for-bit compatible by default.  The
  // standards-compliant normalized Dirichlet mix is enabled explicitly in
  // evaluation experiments; changing it in self-play would change training.
  bool normalized_dirichlet_ = false;
  // Evaluation-only reproducibility switch. Mature training gates keep the
  // historical worker-local RNG stream when false.
  bool deterministic_game_seeds_ = false;
  float fpu_reduction_ = 0.0f;    // first-play-urgency reduction vs parent q
  bool reuse_tree_ = false;       // keep the selected subtree across moves
  int max_retained_nodes_ = 12000; // compact unreachable branches above this
  int max_retained_edges_ = 250000;
  // VCF leaf assist (evaluation-only). 0 disables; >0 runs the one-sided VCF
  // prover with this node budget on sharp leaves (either side has an active
  // four-threat) and pins the proven value (+/-1) instead of the NN value.
  int vcf_leaf_nodes_ = 0;
};

// Single-game MCTS over Gomoku with an injected evaluator. Reused across
// moves of one game via Reset(); not thread-safe.
//
// Sign convention: every value v at a node is the expected outcome from the
// perspective of the player TO MOVE at that node; edge statistics w_/q_ are
// stored from the perspective of the player at the parent node (the one
// choosing the action).
class Mcts {
public:
  struct Edge {
    int action_ = -1;
    int child_ = -1;
    float prior_ = 0.0f;
    float base_prior_ = 0.0f; // network prior before per-root noise
    int n_ = 0;
    float w_ = 0.0f;
  };
  struct Node {
    int edge_begin_ = -1; // index into edges_ (-1: not expanded yet)
    int edge_num_ = 0;
    int n_ = 0;    // total visits through this node
    float w_ = 0.0f; // value sum, perspective of the player to move here
    float q() const { return n_ > 0 ? w_ / n_ : 0.0f; }
  };

  Mcts() : nodes_(), edges_() {
    nodes_.reserve(4096);
    edges_.reserve(4096 * 32);
  }

  // Runs the given number of simulations from the current game position.
  // Fills visit_action / visit_count with per-action visit counts at root.
  void Search(const Gomoku &game, const MctsConfig &config,
              INetEvaluator &evaluator, std::mt19937 &rng,
              std::vector<int> &visit_action, std::vector<int> &visit_count);

  // Moves the persistent root to `action`. Returns true when the matching
  // subtree existed (an unexpanded child is materialized); false resets the
  // tree. Call after every real move when reuse_tree_ is enabled.
  bool AdvanceRoot(int action);
  void Reset();

  bool last_search_reused() const { return last_search_reused_; }
  int root_visits() const;
  std::size_t node_count() const { return nodes_.size(); }
  std::size_t edge_count() const { return edges_.size(); }
  bool budget_exhausted() const { return budget_exhausted_; }
  void RootPriors(std::vector<int> &actions, std::vector<float> &priors) const;

  // Visit-count distribution normalized to a probability vector over all
  // kActionNum actions (zeros elsewhere).
  static void VisitDistribution(const std::vector<int> &visit_action,
                                const std::vector<int> &visit_count,
                                float *pi);

private:
  int AllocateNode();
  // Attaches one edge per legal action of the position to the node.
  void AttachEdges(int node_index, const Gomoku &game, const float *policy);
  // Expands a fresh leaf node with the net. Returns value for the player to
  // move at this node.
  float ExpandNode(int node_index, Gomoku &game, INetEvaluator &evaluator);
  void ExpandRoot(int node_index, Gomoku &game, const MctsConfig &config,
                   INetEvaluator &evaluator, std::mt19937 &rng);
  void ApplyRootNoise(int node_index, const MctsConfig &config,
                      std::mt19937 &rng);
  bool OverRetentionBudget() const;
  void ReleaseTreeStorage();
  static bool SamePosition(const Gomoku &left, const Gomoku &right);
  static float TerminalValue(const Gomoku &game);
  int SelectEdge(int node_index, float c_puct) const;
  // Proves a VCF win/loss for the side to move at this leaf; pins `value`
  // and returns true when decided. Only called when vcf_leaf_nodes_ > 0 and
  // the position is sharp (some side has a live four-threat or five-point).
  bool TryVcfValue(const Gomoku &game, float &value);

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  int root_ = -1;
  Gomoku root_game_;
  bool last_search_reused_ = false;
  int max_retained_nodes_ = 12000;
  int max_retained_edges_ = 250000;
  bool reuse_tree_active_ = false;
  bool budget_exhausted_ = false;
  float fpu_reduction_ = 0.25f; // set at Search() entry from config
  int vcf_leaf_nodes_ = 0;      // set at Search() entry from config
  VcfSolver vcf_solver_;        // reused across leaves; single-threaded
  // scratch
  std::vector<int> path_nodes_;
  std::vector<int> path_edges_;
  std::vector<int> candidate_scratch_;
};

} // namespace az
