#pragma once

#include "game/gomoku.h"

#include <cstdio>
#include <string>

namespace az {

// Bidirectional pipe to one directly exec'ed child process.
class Subprocess {
public:
  Subprocess() = default;
  ~Subprocess();

  bool Start(const char *executable, const char *argument);
  // Writes a line (newline appended) and reads one response line.
  // Returns false on broken pipe/EOF.
  bool Exchange(const std::string &request, std::string &response);
  void Kill();

  bool alive() const { return pid_ > 0; }

private:
  pid_t pid_ = -1;
  FILE *to_child_ = nullptr;
  FILE *from_child_ = nullptr;
};

// Plays moves from the game-old/gobang-web JS AIs (levels 1-7) through a
// persistent Node subprocess. One instance per worker thread.
class JsOpponent {
public:
  // level 1..7 as in the web README difficulty table.
  explicit JsOpponent(int level) : level_(level) {}

  bool Start();
  // Returns the chosen action (cell index), or -1 on failure/illegal move.
  int PickMove(const Gomoku &game);
  void Stop() { process_.Kill(); }

private:
  int level_;
  Subprocess process_;
};

} // namespace az
