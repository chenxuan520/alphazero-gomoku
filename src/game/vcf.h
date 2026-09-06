#pragma once

#include "game/gomoku.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace az {

// One-sided VCF (Victory by Continuous Four) prover for freestyle Gomoku.
//
// Solves: "can `attacker` force a win from this position by an unbroken
// chain of four-threat moves (every attacker move creates at least one new
// immediate five-in-a-row point), against any defense?"  The defender is
// assumed omniscient within the threat space: every reply that removes all
// attacker five-points is tried, and a defender move that completes the
// defender's own five refutes the attack.
//
// Semantics: Solve()=true means proven forced win.  Solve()=false groups
// both "proven not" and "undecided within budget"; callers needing the
// distinction check undecided().
//
// Deterministic, allocation-light, single-threaded; each instance owns its
// transposition table and is reused across calls.
class VcfSolver {
public:
  static constexpr int kBoardCells = Gomoku::kCellNum;

  VcfSolver();

  // att_color: Gomoku::kBlack / kWhite.  node_budget caps explored nodes.
  bool Solve(const std::array<int8_t, kBoardCells> &board, int att_color,
             int node_budget);
  // First move of a proven chain, or -1.
  int FindWinningMove(const std::array<int8_t, kBoardCells> &board,
                      int att_color, int node_budget);

  bool undecided() const { return undecided_; }
  uint64_t nodes() const { return nodes_; }

  // VCT-lite: when enabled, attack generation also includes moves creating a
  // *live three* (a three that next move can become an open four), and the
  // defender may answer a three by covering its two four-squares or by
  // countering with an own four. Still one-sided: proves attacker forcing
  // wins only. Heavier than pure VCF; intended for root probes.
  void set_enable_threes(bool on) { enable_threes_ = on; }

  // --- static pattern helpers (exposed for testing / MCTS leaf gating) ---
  // Empty cells where placing `color` completes five-or-more.
  static void FivePoints(const std::array<int8_t, kBoardCells> &board,
                         int color, std::vector<int> &out);
  // Empty cells near stones (Chebyshev radius) — shared with move-gen order.
  static void NearStones(const std::array<int8_t, kBoardCells> &board,
                         int radius, std::vector<int> &out);
  // Cells where placing `color` creates a live three (next move can create
  // an open four). Slower than FivePoints; used by VCT-lite only.
  static void LiveThreeMoves(const std::array<int8_t, kBoardCells> &board,
                             int color, std::vector<int> &out);

private:
  enum class NodeState : int8_t { kFail = 0, kWin = 1 };

  bool AttackWin(int depth);
  // Assumes attacker stone just placed at `move` having five-points `fp`;
  // tries every defense. Returns true iff attack still wins.
  bool DefenseFails(int depth, const std::vector<int> &fp);
  // Defense replies to a live-three move at `cell`: cover both of the three's
  // four-squares, or counter with an own four. Returns true iff attack wins.
  bool DefenseFailsThree(int depth, const std::vector<int> &covers);
  // Cells where placing attc creates at least one five-point.
  void ThreatMoves(std::vector<int> &out);
  void ThreatMovesFor(int color, std::vector<int> &out);

  bool Place(int cell, int color);
  void Undo(int cell, int color);
  bool WinsAt(int cell, int color) const; // does placing here complete >=5
  static int DirectionRun(const std::array<int8_t, kBoardCells> &board,
                          int row, int col, int dr, int dc, int color);
  // While board has a stone of `color` PRE-PLACED at `cell` virtually:
  // runs over the 4 dirs and appends for each dir the two cells that would
  // extend the current run (first empty at each side within run range).
  static void RunEnds(const std::array<int8_t, kBoardCells> &board, int cell,
                      int color, std::vector<int> &out, int run_at_least);

  std::array<int8_t, kBoardCells> b_{};
  int att_ = 1;
  int def_ = -1;
  int budget_ = 0;
  int iter_budget_used_ = 0;
  bool undecided_ = false;
  bool enable_threes_ = false;
  uint64_t nodes_ = 0;
  int root_move_ = -1;
  std::array<uint64_t, kBoardCells * 2> zobrist_{}; // [cell][black?0:1]
  uint64_t hash_ = 0;
  std::unordered_map<uint64_t, int8_t> tt_;
  std::vector<int> scratch_;
};

} // namespace az
