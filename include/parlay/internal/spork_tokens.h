#ifndef PARLAY_INTERNAL_SPORK_TOKENS_H_
#define PARLAY_INTERNAL_SPORK_TOKENS_H_

// Heartbeat tokens: the interval and what a beat is worth, the running
// computation's balance and credit clock, settle, which converts credit
// into tokens at beats and joins (HeartbeatCarry in spork-proofs2), and
// disabling beats around code that rewrites them.

#include "../portability.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <ctime>

namespace spork {

// simulated heartbeat interval is HEARTBEAT_INTERVAL_US/HEARTBEAT_COARSENING_FACTOR,
// but with exponential work overhead as HEARTBEAT_COARSENING_FACTOR increases
// -DSPORK_HEARTBEAT_INTERVAL_US=N sets it for an experiment.
#ifdef SPORK_HEARTBEAT_INTERVAL_US
inline constexpr unsigned int HEARTBEAT_INTERVAL_US = SPORK_HEARTBEAT_INTERVAL_US;
#else
inline constexpr unsigned int HEARTBEAT_INTERVAL_US = 250;
#endif
// -DSPORK_HEARTBEAT_COARSENING_FACTOR=C sets it for an experiment.
#ifdef SPORK_HEARTBEAT_COARSENING_FACTOR
inline constexpr unsigned int HEARTBEAT_COARSENING_FACTOR = SPORK_HEARTBEAT_COARSENING_FACTOR;
#else
inline constexpr unsigned int HEARTBEAT_COARSENING_FACTOR = 5;
#endif
inline constexpr unsigned int TOKENS_PER_HEARTBEAT = (1 << (1 + HEARTBEAT_COARSENING_FACTOR)) - 1;
// The tokens the running computation may spend on promotions, never more
// than TOKEN_LIMIT (below), and when it last converted its time into
// tokens (task_mark, below):
//  - a heartbeat converts the time since the mark into whole tokens, pro
//    rata (a whole interval is a whole TOKENS_PER_HEARTBEAT), see settle:
//    below the cap the remainder, under one token's worth, stays on the
//    clock, and at the cap it is dropped;
//  - a promotion costs one token, and of the t that remain the left side
//    (the computation that keeps running here) keeps ceil(t/2) and its mark,
//    and the right side (the spawned job) gets floor(t/2), see split_tokens;
//  - the right side starts, stolen or not, with the tokens it was given and
//    its mark at the moment it starts;
//  - at every join, stolen or not, the continuation's tokens and mark come
//    from both sides' balances and elapsed time pooled, converted the same
//    way (settle again), see join_tokens.
inline constinit thread_local volatile unsigned int heartbeat_tokens = 0;
// The cap on a balance.  Beats and joins are the only steps that create
// tokens; both clamp the balance to it and drop the rest of the credit
// (see settle).  By default it is TOKENS_PER_HEARTBEAT.  With
// SPORK_UNCAPPED_TOKENS there is in effect no cap: a beat adds every whole
// token its clock is worth and a join gives every whole token the pooled
// credit is worth, the remainder (under a token's worth) staying on the
// clock either way.  Balances are then still held below 2^30 so that they
// cannot overflow, which a computation would reach only after about an
// hour of sequential running at 63 tokens per 250 us.
#ifdef SPORK_UNCAPPED_TOKENS
inline constexpr unsigned int TOKEN_LIMIT = 1u << 30;
#else
inline constexpr unsigned int TOKEN_LIMIT = TOKENS_PER_HEARTBEAT;
#endif
// Called by a promotion handler once the promotion's token is paid: the
// running computation keeps ceil(t/2) of the t left and the returned
// floor(t/2) are for the job being spawned.
PARLAY_INLINE unsigned int split_tokens() noexcept {
  const unsigned int t = heartbeat_tokens;
  heartbeat_tokens = t - t / 2;
  return t / 2;
}

// ---- time -----------------------------------------------------------------
// A task's elapsed time plays the part of the heartbeat timer in the token
// proofs (spork-proofs2), but the hardware timer is never re-armed: each
// task has a software mark instead, and its elapsed time is now - mark.
// A mark is set when a right side starts (to now), and at the task's beats
// and joins by settle (to now, less the credit not converted).  The credit
// left on a clock is under one token's worth (N / B), but it may reach back
// past this thread's last beat: a negative base in the proofs.  So a task's
// elapsed time exceeds its thread's timer by less than N / B, and a beat
// comes by the time it reaches N + N / B (a missed beat aside): the proofs' L.
//
// The time source is clock_gettime(CLOCK_MONOTONIC): POSIX,
// async-signal-safe, in nanoseconds, and the clock the heartbeat timer
// itself runs on (19 ns a read on the Xeon 8380).  Every join reads it two
// or three times.  rdtsc, 9 ns a read there but needing a TSC that is
// invariant and synchronized across cores, and a calibration, measured the
// same (PBBS at 80 cores 0.998 geomean, clock_gettime / rdtsc; fib 1 core
// +0.1%), and the option to use it was removed.
PARLAY_INLINE std::uint64_t monotonic_ns() noexcept {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000u + static_cast<std::uint64_t>(ts.tv_nsec);
}
inline constexpr std::uint64_t HEARTBEAT_INTERVAL_NS = std::uint64_t{HEARTBEAT_INTERVAL_US} * 1000u;
static_assert(HEARTBEAT_INTERVAL_NS < 1000000000u, "the timer is set through tv_nsec alone");
// With SPORK_NO_CLOCK, time is never converted into tokens or back: there
// are no clocks or marks.  Each beat acted on brings TOKENS_PER_HEARTBEAT
// tokens (up to TOKEN_LIMIT), and a join's continuation gets the two
// sides' balances summed (up to TOKEN_LIMIT), on whatever timer its thread
// has.  The timers run free as always.
#ifdef SPORK_NO_CLOCK
inline constexpr bool NO_CLOCK = true;
#else
inline constexpr bool NO_CLOCK = false;
#endif
// The running computation's mark: saved and restored with its tokens.  Its
// credit clock, now - mark, is the proofs' phase - base: the time it has
// run and not yet converted into tokens.
inline constinit thread_local volatile std::uint64_t task_mark = 0;
// noinline, like fiber_acquire: it is called after joins, which can resume
// a computation on a different thread, so the thread-local is addressed
// afresh on every call.
PARLAY_NOINLINE inline std::uint64_t task_elapsed() noexcept {
  if constexpr (NO_CLOCK) return 0;
  return monotonic_ns() - task_mark;
}
// The same, also storing the time it read in `now`.
PARLAY_NOINLINE inline std::uint64_t task_elapsed(std::uint64_t& now) noexcept {
  if constexpr (NO_CLOCK) { now = 0; return 0; }
  now = monotonic_ns();
  return now - task_mark;
}

// ---- disabling beats -------------------------------------------------------
// true => beats are not acted on: the running computation is reading and
// rewriting its balance or mark, or promoting (see HeartbeatsDisabled)
inline constinit thread_local volatile bool disable_heartbeats = false;
// true => a beat landed while disable_heartbeats was set; HeartbeatsDisabled
// runs it once beats are back on
inline constinit thread_local volatile bool heartbeat_pending = false;

inline void run_pending_heartbeat() noexcept;

// Disables beats for its lifetime, around code that reads and rewrites the
// running computation's balance or mark, or promotes: a beat acted on in
// between would be lost or counted twice.  A beat that lands meanwhile is
// left pending (heartbeat_handler), and the outermost guard runs it on the
// way out, once beats are back on: disabling beats delays a beat by at most
// the guarded code, and loses none.
class HeartbeatsDisabled {
  const bool outermost = !disable_heartbeats;
 public:
  HeartbeatsDisabled() noexcept { disable_heartbeats = true; }
  ~HeartbeatsDisabled() {
    if (!outermost) return;
    disable_heartbeats = false;
    // Check only once enabled: a beat from here on is acted on directly.
    std::atomic_signal_fence(std::memory_order_seq_cst);
    while (heartbeat_pending) [[unlikely]] run_pending_heartbeat();
  }
  HeartbeatsDisabled(const HeartbeatsDisabled&) = delete;
  HeartbeatsDisabled& operator=(const HeartbeatsDisabled&) = delete;
};

// ---- converting credit into tokens ------------------------------------------
// Beats and joins convert credit the same way, by settle, the scheme
// spork-proofs2's HeartbeatCarry.lean proves the work and span bounds for.
// With B = TOKENS_PER_HEARTBEAT and N the interval, a clock c and a balance
// t are together worth credit B * c + N * t.  settle(K) gives the running
// computation s = min(TOKEN_LIMIT, floor(K / N)) tokens.  Below the cap the
// credit not converted, under one token's worth, stays on its clock in
// full, floor((K - N * s) / B) of it, even if that reaches back past this
// thread's last beat (a negative base); at the cap it is dropped and the
// clock restarts at now.  No credit is ever created.
//  - A beat settles the running computation's own credit,
//    K = B * clock + N * t, so below the cap a beat that converts nothing
//    leaves the mark where it was.  (HeartbeatCarry clamps the clock at N
//    here; under the cap that changes nothing, as a clock of N is worth a
//    whole TOKENS_PER_HEARTBEAT by itself.)
//  - A join settles both sides' credit pooled (join_tokens_disabled).
// The hardware timer is never touched.  Called from the beat handler, or
// with beats disabled, so that no beat can land between reading the credit
// and replacing it.
PARLAY_INLINE void settle(std::uint64_t K, std::uint64_t N, std::uint64_t now) noexcept {
  const std::uint64_t s = K / N;
  if (s < TOKEN_LIMIT) {
    heartbeat_tokens = static_cast<unsigned int>(s);
    task_mark = now - (K - N * s) / TOKENS_PER_HEARTBEAT;
  } else {
    heartbeat_tokens = TOKEN_LIMIT;
    task_mark = now;
  }
}

// Every join, stolen or not: c1, t1 are the clock and balance of the side
// that arrived first, c2, t2 those of the side that arrived second, which
// the continuation follows.  Their credit is pooled and settled:
//   K = B * (c1 + c2) + N * (t1 + t2)
// Under a token's worth (K < N) both balances are 0, so the continuation
// gets no tokens and keeps the two sides' clocks together, c1 + c2 (under
// N / B), in full.  The rule is symmetric in the two sides.  Called with
// beats disabled (join_tokens, join_tokens_running).
inline void join_tokens_disabled(std::uint64_t c1, unsigned int t1,
                                 std::uint64_t c2, unsigned int t2, std::uint64_t now) noexcept {
  if constexpr (NO_CLOCK) {   // the balances summed; the clocks are not kept
    heartbeat_tokens = std::min(t1 + t2, TOKEN_LIMIT);
    return;
  }
  const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
  settle(std::uint64_t{TOKENS_PER_HEARTBEAT} * (c1 + c2) + N * (std::uint64_t{t1} + t2), N, now);
}

// A join whose two sides' clocks and balances were recorded beforehand,
// at time `now` if the caller already read it (0: read it here).  A thief
// that arrives second and continues the parent passes the time it read
// for its own clock, a few hundred nanoseconds before on the same thread,
// with nothing else run in between; the join's own time since then counts
// toward the continuation's clock, as all join time does.
// noinline (both entry points): they run after joins, which can resume a
// computation on a different thread, so their thread-locals are addressed
// afresh.
PARLAY_NOINLINE inline void join_tokens(std::uint64_t c1, unsigned int t1,
                                          std::uint64_t c2, unsigned int t2,
                                          std::uint64_t now = 0) noexcept {
  const HeartbeatsDisabled beats_off;
  if (!NO_CLOCK && now == 0) now = monotonic_ns();
  join_tokens_disabled(c1, t1, c2, t2, now);
}

// A join one of whose sides is the computation running on this thread: its
// clock and balance are read here, with beats disabled, rather than by the
// caller, where a beat could land between reading them and this join
// replacing them, and its conversion would then be counted twice or lost.
// Its clock runs up to now.
PARLAY_NOINLINE inline void join_tokens_running(std::uint64_t c_other, unsigned int t_other) noexcept {
  const HeartbeatsDisabled beats_off;
  const std::uint64_t now = NO_CLOCK ? 0 : monotonic_ns();
  const std::uint64_t c = now - task_mark;
  join_tokens_disabled(c_other, t_other, c, heartbeat_tokens, now);
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_TOKENS_H_
