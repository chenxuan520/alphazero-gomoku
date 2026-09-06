#include "game/vcf.h"

#include <algorithm>
#include <random>

namespace az {

namespace {
constexpr int kBoard = Gomoku::kBoardSize;
constexpr int kMaxPly = 64; // VCF chains are far shorter; hard safety cap
} // namespace

VcfSolver::VcfSolver() {
  std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
  for (int i = 0; i < kBoardCells; ++i) {
    zobrist_[2 * i] = rng();
    zobrist_[2 * i + 1] = rng();
  }
  tt_.reserve(1 << 16);
}

int VcfSolver::DirectionRun(const std::array<int8_t, kBoardCells> &board,
                            int row, int col, int dr, int dc, int color) {
  int run = 1;
  for (int s = 1; s < 5; ++s) {
    int nr = row + dr * s, nc = col + dc * s;
    if (nr < 0 || nr >= kBoard || nc < 0 || nc >= kBoard) break;
    if (board[nr * kBoard + nc] != color) break;
    ++run;
  }
  for (int s = 1; s < 5; ++s) {
    int nr = row - dr * s, nc = col - dc * s;
    if (nr < 0 || nr >= kBoard || nc < 0 || nc >= kBoard) break;
    if (board[nr * kBoard + nc] != color) break;
    ++run;
  }
  return run;
}

void VcfSolver::FivePoints(const std::array<int8_t, kBoardCells> &board,
                           int color, std::vector<int> &out) {
  out.clear();
  static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  std::vector<int> cand;
  NearStones(board, 1, cand);
  for (int cell : cand) {
    if (board[cell] != 0) continue;
    int row = cell / kBoard, col = cell % kBoard;
    for (auto &d : kDirs) {
      if (DirectionRun(board, row, col, d[0], d[1], color) >= 5) {
        out.push_back(cell);
        break;
      }
    }
  }
}

void VcfSolver::NearStones(const std::array<int8_t, kBoardCells> &board,
                           int radius, std::vector<int> &out) {
  out.clear();
  bool any_stone = false;
  for (int i = 0; i < kBoardCells; ++i) any_stone |= (board[i] != 0);
  if (!any_stone) {
    out.push_back(kBoardCells / 2);
    return;
  }
  for (int r = 0; r < kBoard; ++r) {
    for (int c = 0; c < kBoard; ++c) {
      if (board[r * kBoard + c] != 0) continue;
      bool near = false;
      for (int dr = -radius; dr <= radius && !near; ++dr) {
        for (int dc = -radius; dc <= radius && !near; ++dc) {
          int nr = r + dr, nc = c + dc;
          if (nr < 0 || nr >= kBoard || nc < 0 || nc >= kBoard) continue;
          near = board[nr * kBoard + nc] != 0;
        }
      }
      if (near) out.push_back(r * kBoard + c);
    }
  }
}

bool VcfSolver::WinsAt(int cell, int color) const {
  static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  int row = cell / kBoard, col = cell % kBoard;
  for (auto &d : kDirs) {
    if (DirectionRun(b_, row, col, d[0], d[1], color) >= 5) return true;
  }
  return false;
}

void VcfSolver::ThreatMovesFor(int color, std::vector<int> &out) {
  out.clear();
  std::vector<int> cand;
  NearStones(b_, 2, cand);
  std::vector<int> fp;
  for (int cell : cand) {
    if (b_[cell] != 0) continue;
    b_[cell] = static_cast<int8_t>(color);
    FivePoints(b_, color, fp);
    if (!fp.empty()) out.push_back(cell);
    b_[cell] = 0;
  }
}

void VcfSolver::ThreatMoves(std::vector<int> &out) { ThreatMovesFor(att_, out); }

// VCT-lite defense against a *live three* just played at cell `move` by the
// attacker: defender may cover any four-extension square of the three
// (covers), or counter with a defender four (which forces the attacker to
// answer next and breaks the forcing chain). The attack survives only if
// every such reply still loses.
bool VcfSolver::DefenseFailsThree(int depth, const std::vector<int> &covers) {
  std::vector<int> cand = covers;
  // Defender's own four-creators are legitimate counter defenses.
  std::vector<int> counter_fours;
  ThreatMovesFor(def_, counter_fours);
  cand.insert(cand.end(), counter_fours.begin(), counter_fours.end());
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::vector<int> fp;
  for (int d : cand) {
    if (b_[d] != 0) continue;
    if (undecided_) return false;
    Place(d, def_);
    bool ok;
    if (WinsAt(d, def_)) {
      ok = true; // defender completes five -> refuted
    } else {
      FivePoints(b_, def_, fp);
      if (!fp.empty()) {
        ok = true; // defender counter-four -> attacker must answer next
      } else {
        ok = !AttackWin(depth + 1);
      }
    }
    Undo(d, def_);
    if (ok) return false; // defense found
  }
  return true;
}

namespace {

// For a stone of `color` at (row,col): for each of the 4 directions returns
// the run length through that cell and the first empty cell beyond each end
// of that run (clipped to the board; -1 when off-board or blocked).
struct RunInfo {
  int len_ = 0;
  int end_a_ = -1; // cell of the first empty square on the +d side
  int end_b_ = -1;
};

RunInfo InspectRun(const std::array<int8_t, Gomoku::kCellNum> &board, int row,
                   int col, int dr, int dc, int color) {
  constexpr int kB = Gomoku::kBoardSize;
  RunInfo info;
  info.len_ = 1;
  auto walk = [&](int sr, int sc, int vr, int vc, int &end_cell) {
    end_cell = -1;
    for (int s = 1; s < 6; ++s) {
      int nr = sr + vr * s, nc = sc + vc * s;
      if (nr < 0 || nr >= kB || nc < 0 || nc >= kB) return;
      int cell = nr * kB + nc;
      if (board[cell] == color) {
        ++info.len_;
        continue;
      }
      if (board[cell] == 0) end_cell = cell;
      return;
    }
  };
  walk(row, col, dr, dc, info.end_a_);
  walk(row, col, -dr, -dc, info.end_b_);
  return info;
}

// Places `color` virtually at `cell` (already verified empty) and appends to
// `covers` each end square that would extend a run to length >= 4.
void ExtendRunCovers(std::array<int8_t, Gomoku::kCellNum> &board, int cell,
                     int color, std::vector<int> &covers, int len_ge) {
  constexpr int kB = Gomoku::kBoardSize;
  static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  int row = cell / kB, col = cell % kB;
  board[cell] = static_cast<int8_t>(color);
  for (auto &d : kDirs) {
    RunInfo info = InspectRun(board, row, col, d[0], d[1], color);
    if (info.len_ >= len_ge) {
      if (info.end_a_ >= 0) covers.push_back(info.end_a_);
      if (info.end_b_ >= 0) covers.push_back(info.end_b_);
    }
  }
  board[cell] = 0;
}

// Same as ExtendRunCovers but assumes the stone of `color` at `cell` was
// already placed (does not place/undo it).
void ExtendRunCoversPlaced(const std::array<int8_t, Gomoku::kCellNum> &board,
                           int cell, int color, std::vector<int> &covers,
                           int len_ge) {
  constexpr int kB = Gomoku::kBoardSize;
  static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  int row = cell / kB, col = cell % kB;
  for (auto &d : kDirs) {
    RunInfo info = InspectRun(board, row, col, d[0], d[1], color);
    if (info.len_ >= len_ge) {
      if (info.end_a_ >= 0) covers.push_back(info.end_a_);
      if (info.end_b_ >= 0) covers.push_back(info.end_b_);
    }
  }
}

} // namespace

void VcfSolver::LiveThreeMoves(
    const std::array<int8_t, kBoardCells> &board, int color,
    std::vector<int> &out) {
  out.clear();
  std::vector<int> cand;
  NearStones(board, 2, cand);
  auto b = board; // local scratch; near-stone cells are legal targets
  std::vector<int> tri_covers, four_covers;
  for (int cell : cand) {
    if (b[cell] != 0) continue;
    b[cell] = static_cast<int8_t>(color); // three stays on the board
    tri_covers.clear();
    ExtendRunCoversPlaced(b, cell, color, tri_covers, 3);
    bool live = false;
    for (int cover : tri_covers) {
      b[cover] = static_cast<int8_t>(color); // now the run actually reaches 4
      four_covers.clear();
      ExtendRunCoversPlaced(b, cover, color, four_covers, 4);
      b[cover] = 0;
      if (!four_covers.empty()) {
        live = true;
        break;
      }
    }
    b[cell] = 0;
    if (live) out.push_back(cell);
  }
}

bool VcfSolver::DefenseFails(int depth, const std::vector<int> &fp) {
  // Any legal reply is a five-point square of the attacker (only those can
  // neutralize the threat).  Tries each; the attack survives only if ALL
  // replies still lose for the defender.
  for (int block : fp) {
    if (b_[block] != 0) continue;
    if (undecided_) return false; // budget stall: attack unproven
    Place(block, def_);
    bool def_made_five = WinsAt(block, def_);
    bool ok = false;
    if (def_made_five) {
      ok = true; // defender completes own five -> attack refuted
    } else {
      std::vector<int> def_fp;
      FivePoints(b_, def_, def_fp);
      if (!def_fp.empty()) {
        ok = true; // defender counter-creates an immediate five threat; the
                   // forcing chain is broken (attacker must answer next)
      } else {
        std::vector<int> next_fp;
        FivePoints(b_, att_, next_fp);
        if (next_fp.empty()) {
          // threat neutralized; attack continues deeper
          ok = !AttackWin(depth + 1);
        } // else: defender failed to cover all -> attack wins this branch
      }
    }
    Undo(block, def_);
    if (ok) return false; // found saving defense
  }
  return true;
}

bool VcfSolver::AttackWin(int depth) {
  if (depth > kMaxPly) return false;
  if (++nodes_ > static_cast<uint64_t>(budget_)) {
    undecided_ = true;
    return false;
  }
  auto it = tt_.find(hash_);
  if (it != tt_.end()) return it->second == static_cast<int8_t>(NodeState::kWin);

  // Existing immediate five: take it.
  std::vector<int> fp;
  FivePoints(b_, att_, fp);
  if (!fp.empty()) {
    tt_[hash_] = static_cast<int8_t>(NodeState::kWin);
    if (depth == 0) root_move_ = fp[0];
    return true;
  }

  std::vector<int> threats;
  ThreatMoves(threats);
  bool win = false;
  for (int m : threats) {
    Place(m, att_);
    bool made_five = WinsAt(m, att_);
    if (made_five) {
      win = true;
    } else {
      std::vector<int> new_fp;
      FivePoints(b_, att_, new_fp);
      win = !new_fp.empty() && DefenseFails(depth, new_fp);
    }
    if (win && depth == 0) root_move_ = m;
    Undo(m, att_);
    if (win) break;
  }
  // VCT-lite: live-three creators (forcing chains through threes).
  if (!win && enable_threes_) {
    std::vector<int> threes;
    LiveThreeMoves(b_, att_, threes);
    for (int m : threes) {
      Place(m, att_);
      std::vector<int> new_fp;
      FivePoints(b_, att_, new_fp);
      if (!new_fp.empty()) {
        // also a four-creator; already handled above
        Undo(m, att_);
        continue;
      }
      std::vector<int> covers;
      ExtendRunCovers(b_, m, att_, covers, 3);
      win = !covers.empty() && DefenseFailsThree(depth, covers);
      if (win && depth == 0) root_move_ = m;
      Undo(m, att_);
      if (win) break;
    }
  }
  if (!undecided_) tt_[hash_] = static_cast<int8_t>(win ? NodeState::kWin
                                                        : NodeState::kFail);
  return win;
}

bool VcfSolver::Solve(const std::array<int8_t, kBoardCells> &board,
                      int att_color, int node_budget) {
  b_ = board;
  att_ = att_color;
  def_ = -att_color;
  budget_ = node_budget;
  undecided_ = false;
  nodes_ = 0;
  root_move_ = -1;
  hash_ = 0;
  for (int i = 0; i < kBoardCells; ++i) {
    if (b_[i] != 0) hash_ ^= zobrist_[2 * i + (b_[i] == Gomoku::kBlack ? 0 : 1)];
  }
  tt_.clear();
  return AttackWin(0);
}

int VcfSolver::FindWinningMove(const std::array<int8_t, kBoardCells> &board,
                               int att_color, int node_budget) {
  return Solve(board, att_color, node_budget) ? root_move_ : -1;
}

bool VcfSolver::Place(int cell, int color) {
  b_[cell] = static_cast<int8_t>(color);
  hash_ ^= zobrist_[2 * cell + (color == Gomoku::kBlack ? 0 : 1)];
  return true;
}

void VcfSolver::Undo(int cell, int color) {
  hash_ ^= zobrist_[2 * cell + (color == Gomoku::kBlack ? 0 : 1)];
  b_[cell] = 0;
}

} // namespace az
