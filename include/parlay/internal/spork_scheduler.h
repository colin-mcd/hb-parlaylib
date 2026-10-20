#ifndef PARLAY_INTERNAL_SPORK_SCHEDULER_H_
#define PARLAY_INTERNAL_SPORK_SCHEDULER_H_

#include "work_stealing_deque.h"
#include "spork_fiber.h"
#include "../monoid.h"
#include "../portability.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <ctime>
#ifdef SPORK_HEARTBEAT_TIMING
#include <x86intrin.h>   // __rdtsc, for the heartbeat timing counters
#endif
#include <iostream>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define USE_SIGNAL_SAFE_ATOMIC
#define RECORD_HEARTBEAT_STATS

namespace parlay {
  template <typename Job>
  class scheduler;
  namespace internal {
    unsigned int init_num_workers();
  }
}

namespace spork {

  // simulated heartbeat interval is HEARTBEAT_INTERVAL_US/HEARTBEAT_COARSENING_FACTOR,
  // but with exponential work overhead as HEARTBEAT_COARSENING_FACTOR increases
  inline constexpr unsigned int HEARTBEAT_INTERVAL_US = 250;
  inline constexpr unsigned int HEARTBEAT_COARSENING_FACTOR = 6;
  inline constexpr unsigned int TOKENS_PER_HEARTBEAT = (1 << HEARTBEAT_COARSENING_FACTOR) - 1;
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
  // and joins by settle (to now, less the credit not converted); with
  // SPORK_IGNORE_HB_WORK it is also moved past the time spent promoting and
  // joining (see IGNORE_HB_WORK).  The credit left on a clock is under one
  // token's worth (N / B), but it may reach back past this thread's last
  // beat: a negative base in the proofs.  So a task's elapsed time exceeds
  // its thread's timer by less than N / B, and a beat comes by the time it
  // reaches N + N / B (a missed beat aside): the proofs' L.
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
  // The running computation's mark: saved and restored with its tokens.  Its
  // credit clock, now - mark, is the proofs' phase - base: the time it has
  // run and not yet converted into tokens.
  inline constinit thread_local volatile std::uint64_t task_mark = 0;
  // noinline, like fiber_acquire: it is called after joins, which can resume
  // a computation on a different thread, so the thread-local is addressed
  // afresh on every call.
  PARLAY_NOINLINE inline std::uint64_t task_elapsed() noexcept {
    return monotonic_ns() - task_mark;
  }
  // The same, also storing the time it read in `now`.
  PARLAY_NOINLINE inline std::uint64_t task_elapsed(std::uint64_t& now) noexcept {
    now = monotonic_ns();
    return now - task_mark;
  }

  // ---- heartbeat work and the clock -------------------------------------------
  // By default the time a computation spends promoting and joining counts
  // towards its clock like any other.  With SPORK_IGNORE_HB_WORK it does not,
  // as in the proofs, where these take no time: the mark is moved past each
  // promotion (in a beat's handler, or eager) and past each join's
  // bookkeeping (popping or waiting for the job, joining the tokens), and a
  // parent's clock stops when it arrives at the join.  A stolen job's clock
  // starts just before the job runs.
#ifdef SPORK_IGNORE_HB_WORK
  inline constexpr bool IGNORE_HB_WORK = true;
#else
  inline constexpr bool IGNORE_HB_WORK = false;
#endif
  // The running computation's clock as of time t, read earlier with beats
  // enabled: a beat since may have moved the mark past t, in which case the
  // time up to t has already been converted and the clock is 0.
  PARLAY_INLINE std::uint64_t clock_at(std::uint64_t t) noexcept {
    const std::uint64_t mark = task_mark;
    return t > mark ? t - mark : 0;
  }
  // true => this *job* must not be promoted; used during eager promotions to prevent signal-delivered heartbeats from promoting simultaneously
  inline constinit thread_local volatile bool disable_heartbeats = false;
  // true => this *processor* (= thread) is between jobs; not fiber state
  inline constinit thread_local volatile bool heartbeats_idle = true;
  // true => the *scheduler* is done, e.g., at end of program
  inline constinit std::atomic<bool> heartbeats_done = false;
  // true => this thread's heartbeat handler is running.  The handler is
  // installed SA_NODEFER (start_heartbeats), so SIGALRM stays unblocked while
  // it runs: the thread's signal mask then never changes on the way into the
  // handler or out of it, and the kernel skips taking the process-wide signal
  // lock for those two changes.  A beat could then interrupt the handler
  // itself, though only one held up a whole interval; it finds this set and
  // is dropped.
  inline constinit thread_local volatile bool in_heartbeat_handler = false;

#ifdef SPORK_HEARTBEAT_TIMING
  // ---- measuring what heartbeats cost (-DSPORK_HEARTBEAT_TIMING) -------------
  // Off by default.  Each thread counts, in TSC ticks, the signals it took
  // while running a computation (busy) or between jobs (idle), how long its
  // handler ran, from entry to exit, and how long it spent between jobs; and
  // the promotions it made, eager or in the handler, and how long its eager
  // promotions took (eager_promote, entry to exit).  The kernel's part of a
  // signal (the timer interrupt, delivering the signal, returning from it) is
  // not seen from here.  pbbsbench/common/time_loop.h prints the totals for
  // each round (heartbeat_timing_report).
#if !defined(__x86_64__)
#error "SPORK_HEARTBEAT_TIMING reads the TSC"
#endif
  struct alignas(64) HeartbeatTiming {
    std::uint64_t beats_busy = 0;
    std::uint64_t beats_idle = 0;
    std::uint64_t handler_ticks_busy = 0;
    std::uint64_t handler_ticks_idle = 0;
    std::uint64_t promotions = 0;
    std::uint64_t handler_promotions = 0;
    std::uint64_t eager_ticks = 0;         // time in eager_promote, entry to exit
    std::uint64_t joins = 0;               // syncs of promoted forks
    std::uint64_t stolen_joins = 0;        // those whose job was stolen
    std::uint64_t idle_ticks = 0;          // closed idle periods
    std::uint64_t idle_since = 0;          // start of the open idle period, or 0
  };
  inline constexpr unsigned int HEARTBEAT_TIMING_MAX_THREADS = 4096;
  inline HeartbeatTiming* heartbeat_timing_all[HEARTBEAT_TIMING_MAX_THREADS];
  inline std::atomic<unsigned int> heartbeat_timing_count{0};
  inline constinit thread_local HeartbeatTiming* heartbeat_timing_mine = nullptr;

  // A counter is written only by its own thread, with relaxed atomic stores,
  // so that a snapshot taken on another thread reads whole values.
  PARLAY_INLINE void timing_add(std::uint64_t& c, std::uint64_t d) noexcept {
    __atomic_store_n(&c, c + d, __ATOMIC_RELAXED);
  }

  // Called once per thread, from start_heartbeats, before its timer is armed.
  inline void heartbeat_timing_register() {
    HeartbeatTiming* t = new HeartbeatTiming;   // never freed: snapshots may read it at any time
    heartbeat_timing_mine = t;
    const unsigned int k = heartbeat_timing_count.fetch_add(1, std::memory_order_relaxed);
    if (k < HEARTBEAT_TIMING_MAX_THREADS) __atomic_store_n(&heartbeat_timing_all[k], t, __ATOMIC_RELEASE);
  }

  struct HeartbeatTimingTotals {
    std::uint64_t ns = 0, tsc = 0;
    unsigned int threads = 0;
    std::uint64_t beats_busy = 0, beats_idle = 0, handler_ticks_busy = 0, handler_ticks_idle = 0,
                  promotions = 0, handler_promotions = 0, eager_ticks = 0, idle_ticks = 0,
                  joins = 0, stolen_joins = 0;
  };

  // Every thread's counters summed, an open idle period counted up to now.
  inline HeartbeatTimingTotals heartbeat_timing_snapshot() noexcept {
    HeartbeatTimingTotals s;
    s.ns = monotonic_ns();
    s.tsc = __rdtsc();
    const unsigned int n = std::min(heartbeat_timing_count.load(std::memory_order_acquire),
                                    HEARTBEAT_TIMING_MAX_THREADS);
    for (unsigned int k = 0; k < n; k++) {
      const HeartbeatTiming* t = __atomic_load_n(&heartbeat_timing_all[k], __ATOMIC_ACQUIRE);
      if (t == nullptr) continue;
      s.threads++;
      s.beats_busy += __atomic_load_n(&t->beats_busy, __ATOMIC_RELAXED);
      s.beats_idle += __atomic_load_n(&t->beats_idle, __ATOMIC_RELAXED);
      s.handler_ticks_busy += __atomic_load_n(&t->handler_ticks_busy, __ATOMIC_RELAXED);
      s.handler_ticks_idle += __atomic_load_n(&t->handler_ticks_idle, __ATOMIC_RELAXED);
      s.promotions += __atomic_load_n(&t->promotions, __ATOMIC_RELAXED);
      s.handler_promotions += __atomic_load_n(&t->handler_promotions, __ATOMIC_RELAXED);
      s.eager_ticks += __atomic_load_n(&t->eager_ticks, __ATOMIC_RELAXED);
      s.joins += __atomic_load_n(&t->joins, __ATOMIC_RELAXED);
      s.stolen_joins += __atomic_load_n(&t->stolen_joins, __ATOMIC_RELAXED);
      const std::uint64_t since = __atomic_load_n(&t->idle_since, __ATOMIC_RELAXED);
      s.idle_ticks += __atomic_load_n(&t->idle_ticks, __ATOMIC_RELAXED) +
                      (since != 0 && s.tsc > since ? s.tsc - since : 0);
    }
    return s;
  }

  // One line with the differences between two snapshots.
  inline void heartbeat_timing_report(const HeartbeatTimingTotals& a, const HeartbeatTimingTotals& b) {
    std::printf("Spork heartbeat timing: threads %u ns %llu tsc %llu beats_busy %llu beats_idle %llu "
                "handler_ticks_busy %llu handler_ticks_idle %llu promotions %llu handler_promotions %llu "
                "eager_ticks %llu idle_ticks %llu joins %llu stolen_joins %llu\n",
                b.threads, (unsigned long long)(b.ns - a.ns), (unsigned long long)(b.tsc - a.tsc),
                (unsigned long long)(b.beats_busy - a.beats_busy), (unsigned long long)(b.beats_idle - a.beats_idle),
                (unsigned long long)(b.handler_ticks_busy - a.handler_ticks_busy),
                (unsigned long long)(b.handler_ticks_idle - a.handler_ticks_idle),
                (unsigned long long)(b.promotions - a.promotions),
                (unsigned long long)(b.handler_promotions - a.handler_promotions),
                (unsigned long long)(b.eager_ticks - a.eager_ticks),
                (unsigned long long)(b.idle_ticks - a.idle_ticks),
                (unsigned long long)(b.joins - a.joins), (unsigned long long)(b.stolen_joins - a.stolen_joins));
    std::fflush(stdout);
  }
#endif

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
    const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
    settle(std::uint64_t{TOKENS_PER_HEARTBEAT} * (c1 + c2) + N * (std::uint64_t{t1} + t2), N, now);
  }

  // A join whose two sides' clocks and balances were recorded beforehand,
  // at time `now` if the caller already read it (0: read it here).  A thief
  // that arrives second and continues the parent passes the time it read
  // for its own clock, a few hundred nanoseconds before on the same thread,
  // with nothing else run in between; the join's own time since then counts
  // toward the continuation's clock, as all join time does by default.
  // noinline (both entry points): they run after joins, which can resume a
  // computation on a different thread, so their thread-locals are addressed
  // afresh.
  PARLAY_NOINLINE inline void join_tokens(std::uint64_t c1, unsigned int t1,
                                            std::uint64_t c2, unsigned int t2,
                                            std::uint64_t now = 0) noexcept {
    const bool disabled = disable_heartbeats;
    disable_heartbeats = true;
    if (now == 0) now = monotonic_ns();
    join_tokens_disabled(c1, t1, c2, t2, now);
    if constexpr (IGNORE_HB_WORK) task_mark = task_mark + (monotonic_ns() - now);   // the join's own time
    disable_heartbeats = disabled;
  }

  // A join one of whose sides is the computation running on this thread: its
  // clock and balance are read here, with beats disabled, rather than by the
  // caller, where a beat could land between reading them and this join
  // replacing them, and its conversion would then be counted twice or lost.
  // Its clock runs up to now, or, given arrived (with SPORK_IGNORE_HB_WORK),
  // up to its arrival at the join.
  PARLAY_NOINLINE inline void join_tokens_running(std::uint64_t c_other, unsigned int t_other,
                                                    std::uint64_t arrived = 0) noexcept {
    const bool disabled = disable_heartbeats;
    disable_heartbeats = true;
    const std::uint64_t now = monotonic_ns();
    const std::uint64_t c = (IGNORE_HB_WORK && arrived) ? clock_at(arrived) : now - task_mark;
    join_tokens_disabled(c_other, t_other, c, heartbeat_tokens, now);
    if constexpr (IGNORE_HB_WORK) task_mark = task_mark + (monotonic_ns() - now);   // the join's own time
    disable_heartbeats = disabled;
  }

  // ---- patient stealing from the child at a pinned join ---------------------
  // A join on a pinned stack (the main thread's) cannot hand its continuation
  // to the thief, so the parent waits, helping with other work meanwhile.  A
  // job stolen at random can outlast the child and delay everything after the
  // join, so each search for work first tries the deque of the worker running
  // the child, JOIN_THIEF_TRIES times, JOIN_THIEF_DELAY_NS apart: the child's
  // own promoted work is found there, and running it cannot delay the join.
  // As in hb3-parlaylib.  20 tries measured the same as 250 and as none
  // (BFS on the 3D grid perhaps 4% faster, within noise).
  // SPORK_JOIN_THIEF_TRIES and SPORK_JOIN_THIEF_DELAY_NS override, and
  // SPORK_JOIN_THIEF_TRIES=0 steals at random at once.
  inline unsigned int JOIN_THIEF_TRIES = 20;
  inline unsigned int JOIN_THIEF_DELAY_NS = 1000;
  inline unsigned int join_thief_tries() noexcept { return JOIN_THIEF_TRIES; }
  inline unsigned int join_thief_delay_ns() noexcept { return JOIN_THIEF_DELAY_NS; }

  // ---- stack handoff, thread state ---------------------------------------
  struct Fiber;
  // The fiber this thread is currently running, or null when it is running on
  // its own stack (the main thread's, or a worker's idle loop).
  inline constinit thread_local Fiber* current_fiber = nullptr;
  // Where a fiber goes when it has nothing more to run on this thread.
  // This thread's home handle while it runs a fiber: one-shot, consumed by
  // whichever context jumps back to the idle loop.
  inline constinit thread_local fcontext_t fiber_home_ctx = nullptr;
  // A fiber abandoned by whoever last switched to us, waiting to be recycled.
  inline constinit thread_local Fiber* fiber_pending_free = nullptr;
  // A join whose counter must be decremented once we are off the stack that
  // the decrement makes available to the other arriver.
  inline constinit thread_local WorkStealingJob* fiber_post_join = nullptr;
  // Set while the thread-local view of the computation does not match the
  // stack in use.  Read by the heartbeat handler.
  inline constinit thread_local volatile bool fiber_switching = false;

  // Only a fiber can be handed to another thread.  A thread's own stack is
  // pinned, because the frames beneath the parallel region belong to that
  // thread: main(), or the scheduler's worker loop.
  inline bool handoff_possible() noexcept { return current_fiber != nullptr; }

inline void start_heartbeats() noexcept;
inline void pause_heartbeats() noexcept;
inline void stop_heartbeats() noexcept;

struct WorkStealingJob {
  using scheduler_t = parlay::scheduler<WorkStealingJob>;
  static scheduler_t& get_current_scheduler() {
    scheduler_t* current_scheduler = scheduler_t::get_current_scheduler();
    if (current_scheduler == nullptr) {
      static thread_local scheduler_t local_scheduler(parlay::internal::init_num_workers());
      return local_scheduler;
    }
    return *current_scheduler;
  }
  static unsigned int worker_id() {
    return get_current_scheduler().worker_id();
  }
  static unsigned int num_workers() {
    return get_current_scheduler().num_workers();
  }

  WorkStealingJob() {}

  void operator()() {
    // This job may run nested inside a join's wait(), on a thread that already
    // holds tokens belonging to the frame that is waiting.  Run on the tokens
    // this job was given, leave what it has left for its join, and put the
    // enclosing frame's back on the way out.  Both boundaries run with
    // heartbeats paused, so no beat can be lost in the window.
    const unsigned int enclosing = heartbeat_tokens;
    const std::uint64_t enclosing_mark = task_mark;
    heartbeat_tokens = tokens;
    task_mark = monotonic_ns();
    thief = worker_id();   // lets a parent waiting at a pinned join steal from where the child is
    start_heartbeats();
    run();
    pause_heartbeats();
    tokens = heartbeat_tokens;
    child_progress = task_elapsed();
    heartbeat_tokens = enclosing;
    task_mark = enclosing_mark;
    bool was_done = done.test_and_set(std::memory_order_release);
    assert(!was_done);
  }

  [[nodiscard]] bool finished() volatile const noexcept {
    return done.test(std::memory_order_acquire);
  }

  void wait() const noexcept {
    pause_heartbeats();
    get_current_scheduler().wait_for_job(this);
    start_heartbeats();
  }

  // Spawn this job with t tokens of its own (from split_tokens).  Through
  // the deque cached for this thread; the scheduler is looked up (and made,
  // if need be) only on a thread that has none yet.
  void enqueue(unsigned int t) {
    tokens = t;
    handoff = handoff_possible();
    if (handoff) join_count.store(2, std::memory_order_relaxed);
    thief = NO_THIEF;
    if (!scheduler_t::spawn_here(this)) [[unlikely]] get_current_scheduler().spawn(this);
  }

  static bool try_dequeue() {
    if (auto* d = scheduler_t::own_deque()) [[likely]] return d->pop_bottom() != nullptr;
    return get_current_scheduler().get_own_job() != nullptr;
  }

  // Run by the parent, which finds the job not stolen.  It is a join like
  // any other: the parent (left side) has finished, with its elapsed time
  // and balance; the job (right side) starts with a fresh mark and the
  // tokens it was given, as a thief would start it; and once it finishes,
  // join_tokens pools the two.
  void fast_clone(std::uint64_t arrived) {
    // With beats disabled, so that no beat can land between reading the
    // parent's clock and balance and replacing them with the job's.  The
    // parent's clock runs up to now, or with SPORK_IGNORE_HB_WORK up to its
    // arrival at the join; the job's starts now either way.
    const bool disabled = disable_heartbeats;
    disable_heartbeats = true;
    const std::uint64_t now = monotonic_ns();
    const std::uint64_t e1 = IGNORE_HB_WORK ? clock_at(arrived) : now - task_mark;
    const unsigned int t1 = heartbeat_tokens;
    task_mark = now;
    heartbeat_tokens = tokens;
    disable_heartbeats = disabled;
    const_cast<WorkStealingJob*>(this)->run();
    join_tokens_running(e1, t1);   // the job, still running here, is the second side
  }

  // Defined below, once fibers are available.
  void sync();

  bool sync_is_stolen() {
    if (!try_dequeue()) {
      if (!finished()) wait();
      return true;
    }
    return false;
  }

  virtual void run() = 0;
  volatile std::atomic_flag done;
  volatile unsigned int thief;        // worker running this job, once one has started it
  // The tokens the job starts with, set by enqueue; once it has run, the
  // balance it finished with, for a join at which it arrives second.
  unsigned int tokens;
  // Once stolen, each side's elapsed time since its mark (task_elapsed) when
  // it arrived at the join: the thief's when the job finished, and the
  // parent's when it left its stack at a handoff join.
  std::uint64_t child_progress;
  std::uint64_t parent_progress;
  // When the thief read the time for child_progress: the join's time if the
  // thief arrives second and continues the parent (see join_tokens).
  std::uint64_t child_end;

  // ---- stack handoff -----------------------------------------------------
  // Set when this job is enqueued from a stack that can be suspended (that
  // is, from a fiber).  Jobs enqueued from a pinned stack, which is any
  // thread's own stack, keep the older discipline: the parent waits and the
  // parent resumes.
  bool handoff;
  // Counts down from two: the parent arriving at the join and the thief
  // finishing the job each decrement it once.  Whichever decrement reads one
  // belongs to the second arriver, who continues the parent's continuation.
  std::atomic<int> join_count;
  // The parent's continuation.  It exists only once the parent has left that
  // stack, because an fcontext handle is produced by the jump that vacates it.
  fcontext_t parent_ctx;
};

  template <typename T>
  class async_signal_safe_pointer {
    static_assert(__atomic_always_lock_free(sizeof(T*), 0),
                  "async_signal_safe_pointer relies on being always lock free!");
    private:
    // Accessed only atomically (relaxed, behind signal fences), but a plain
    // pointer rather than a std::atomic, which C++20 always initializes: a
    // spork slot's next is left unwritten (no_init) until it is needed.
    T* ptr;
    public:
    struct no_init_t {};

    inline void store(T* p) noexcept {
      std::atomic_signal_fence(std::memory_order_release);
      __atomic_store_n(&ptr, p, __ATOMIC_RELAXED);
    }

    inline T* load() const noexcept {
      T* p = __atomic_load_n(&ptr, __ATOMIC_RELAXED);
      std::atomic_signal_fence(std::memory_order_acquire);
      return p;
    }

    inline constexpr async_signal_safe_pointer() noexcept : ptr(nullptr) {}

    inline explicit async_signal_safe_pointer(no_init_t) noexcept {}

    inline async_signal_safe_pointer(T* p) noexcept {
      store(p);
    }

    inline T& operator*() {
      return *load();
    }

    inline const T& operator*() const {
      return *load();
    }

    inline T* operator->() noexcept {
      return load();
    }

    inline const T* operator->() const noexcept {
      return load();
    }

    inline async_signal_safe_pointer<T>& operator=(async_signal_safe_pointer<T>&& other) noexcept {
      store(other.load());
      return *this;
    }

    inline bool operator==(async_signal_safe_pointer<T>&& other) const noexcept {
      return load() == other.load();
    }
  };

  // A promotable fork: pushed onto its stack's list for the duration of the
  // fork's left side (spork_slot), oldest first.  The fork that owns it
  // derives from it, with whatever its promotion needs (see spork and par).
  struct SporkSlot {
    volatile bool promoted; // TODO: change this to an integer, record the stealing thread? or maybe needs to be separate
    // The promotion, called with this slot.  A plain function, which knows
    // what the slot belongs to, rather than a closure object: pushing a slot
    // writes one pointer for it.
    void (*promfn)(SporkSlot*);
    async_signal_safe_pointer<SporkSlot>* prev;
    // The next newer slot, if any.  Left unwritten until that slot links in:
    // promote_front reads it only after checking spork_deque_back, which is
    // this very field until then.
    async_signal_safe_pointer<SporkSlot> next;

    // A list's sentinel (spork_deque_front, Fiber::deque_front).
    constexpr explicit SporkSlot() :
      promoted(true), promfn(nullptr), prev(nullptr), next(async_signal_safe_pointer<SporkSlot>()) {}

    // Not yet on the list: its owner finishes building it, and spork_slot
    // links it.
    explicit SporkSlot(void (*fn)(SporkSlot*)) noexcept :
      promoted(false), promfn(fn), next(async_signal_safe_pointer<SporkSlot>::no_init_t{}) {}

    void link() noexcept;
    static void reset();
    bool close();
    void promote();
    static void promote_front();
    void eager_promote();
  };

  // Slots live in the frames of the stack that created them, so the list
  // head belongs to that stack too: a fiber owns its own sentinel, and
  // spork_deque_front is the sentinel for a thread's own (pinned) stack.
  // spork_deque_front_p names whichever is in use on this thread right now,
  // and both it and the back pointer are saved and restored across a handoff.
  inline constinit thread_local SporkSlot spork_deque_front{};
  inline constinit thread_local SporkSlot* spork_deque_front_p = nullptr;
  inline constinit thread_local async_signal_safe_pointer<async_signal_safe_pointer<SporkSlot>> spork_deque_back;

  // Make this the newest slot.  Everything its promotion reads must be set
  // first: from here on the heartbeat handler may promote it.
  inline void SporkSlot::link() noexcept {
    prev = spork_deque_back.load();
    *prev = this;
    spork_deque_back = &next;
  }

  inline void SporkSlot::reset() {
    spork_deque_back = &spork_deque_front_p->next;
  }

  inline bool SporkSlot::close() {
    spork_deque_back = prev;
    return promoted;
  }

  inline void SporkSlot::promote() {
#ifdef SPORK_HEARTBEAT_TIMING
    if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->promotions, 1);
#endif
    heartbeat_tokens = heartbeat_tokens - 1;
    promoted = true;
    promfn(this);
  }

  inline void SporkSlot::promote_front() {
    if (&spork_deque_front_p->next == spork_deque_back) return;
    SporkSlot* slot = spork_deque_front_p->next.load();
    while (heartbeat_tokens) {
      if (!slot->promoted) slot->promote();
      if (&slot->next == spork_deque_back) break;
      slot = slot->next.load();
    }
  }

  // ---- fibers -------------------------------------------------------------
  //
  // Every job runs on a stack of its own, so that a join can suspend it.  The
  // Fiber object sits at the top of the stack it describes; the stack grows
  // down from just below it towards a guard page.

  struct Fiber {
    void* map_base;
    void* stack_top;
    WorkStealingJob* job;
    Fiber* next_free;
    // This stack's computation state while it is not running.
    async_signal_safe_pointer<SporkSlot>* saved_back;
    unsigned int saved_tokens;
    std::uint64_t saved_mark;
    bool saved_disable;
    fcontext_t entry;        // enters fiber_trampoline on this stack
    std::size_t stack_size;      // context that enters fiber_body for the first time
    SporkSlot deque_front;      // head of this stack's spork deque
  };

  [[noreturn]] void fiber_body(Fiber* f);

  // Free fibers.  Each thread caches up to FIBER_CACHE_MAX of them.  A thread
  // whose cache overflows moves FIBER_BATCH to a shared pool, and one whose
  // cache is empty takes up to FIBER_BATCH back before it maps a new stack.
  // A fiber is freed on whichever thread ends up leaving it, not the one that
  // acquired it, so with purely per-thread lists some threads hoard stacks
  // while others keep mapping new ones: three rounds of gbbs BFS on the 3D
  // grid mapped 52,000 stacks for a peak of 187 live.  The pool's lock is
  // taken only to move a batch, and never from the heartbeat handler.
  //
  // Both functions are noinline so that the thread-local lists are addressed
  // afresh on every call: they run after context switches, which can resume a
  // computation on a different thread.
  inline constexpr unsigned int FIBER_CACHE_MAX = 16;
  inline constexpr unsigned int FIBER_BATCH = FIBER_CACHE_MAX / 2;
  inline constinit thread_local Fiber* fiber_free_list = nullptr;
  inline constinit thread_local unsigned int fiber_free_count = 0;

  struct FiberPool {
    std::mutex lock;
    Fiber* head = nullptr;
  };
  inline FiberPool fiber_pool;

  PARLAY_NOINLINE inline Fiber* fiber_acquire() {
    if (fiber_free_list == nullptr) {
      std::lock_guard<std::mutex> g(fiber_pool.lock);
      if (Fiber* h = fiber_pool.head) {
        Fiber* t = h;
        unsigned int n = 1;
        while (n < FIBER_BATCH && t->next_free != nullptr) { t = t->next_free; n++; }
        fiber_pool.head = t->next_free;
        t->next_free = nullptr;
        fiber_free_list = h;
        fiber_free_count = n;
      }
    }
    if (Fiber* f = fiber_free_list) {
      fiber_free_list = f->next_free;
      fiber_free_count--;
      return f;
    }
    void* base = fiber_map_stack();
    auto top = reinterpret_cast<std::uintptr_t>(base) + FIBER_STACK_SIZE - sizeof(Fiber);
    // default-initialised: SporkSlot's default constructor is explicit
    Fiber* f = new (reinterpret_cast<void*>(top & ~std::uintptr_t{63})) Fiber;
    f->map_base = base;
    f->stack_size = reinterpret_cast<std::uintptr_t>(f) - reinterpret_cast<std::uintptr_t>(base) - FIBER_GUARD_SIZE;
    f->stack_top = reinterpret_cast<void*>((top & ~std::uintptr_t{63}) & ~std::uintptr_t{15});
    return f;
  }

  PARLAY_NOINLINE inline void fiber_release(Fiber* f) noexcept {
    f->next_free = fiber_free_list;
    fiber_free_list = f;
    if (++fiber_free_count > FIBER_CACHE_MAX) {
      Fiber* h = fiber_free_list;
      Fiber* t = h;
      for (unsigned int i = 1; i < FIBER_BATCH; i++) t = t->next_free;
      fiber_free_list = t->next_free;
      fiber_free_count -= FIBER_BATCH;
      std::lock_guard<std::mutex> g(fiber_pool.lock);
      t->next_free = fiber_pool.head;
      fiber_pool.head = h;
    }
  }

  void fiber_trampoline(transfer_t t);

  inline void fiber_prepare(Fiber* f) noexcept {
    f->entry = make_fcontext(f->stack_top, f->stack_size, fiber_trampoline);
  }

  // A thread's own stack gets an empty spork deque of its own.
  inline void spork_thread_init() noexcept {
    if (spork_deque_front_p == nullptr) {
      spork_deque_front_p = &spork_deque_front;
      SporkSlot::reset();
    }
  }

  // Make f's computation the one this thread is running.
  inline void fiber_install(Fiber* f) noexcept {
    current_fiber = f;
    spork_deque_front_p = &f->deque_front;
    spork_deque_back = f->saved_back;
    heartbeat_tokens = f->saved_tokens;
    task_mark = f->saved_mark;
    disable_heartbeats = f->saved_disable;
  }

  inline void fiber_save(Fiber* f) noexcept {
    f->saved_back = spork_deque_back.load();
    f->saved_tokens = heartbeat_tokens;
    f->saved_mark = task_mark;
    f->saved_disable = disable_heartbeats;
  }

  // Recycle the fiber abandoned by whoever switched to us.  At most one can be
  // waiting, because a fiber is abandoned only immediately before a switch.
  inline void fiber_drain() noexcept {
    if (fiber_pending_free) { fiber_release(fiber_pending_free); fiber_pending_free = nullptr; }
  }

  // Boost hands an entering context a handle to whoever jumped in, which is
  // this thread's idle loop: where this fiber goes when it is done.
  inline void fiber_trampoline(transfer_t t) {
    fiber_home_ctx = t.fctx;
    fiber_body(static_cast<Fiber*>(t.data));
  }

  [[noreturn]] inline void fiber_body(Fiber* f) {
    WorkStealingJob* job = f->job;
    f->deque_front.promoted = true;
    f->deque_front.next.store(nullptr);
    f->saved_back = &f->deque_front.next;
    f->saved_tokens = job->tokens;
    f->saved_mark = monotonic_ns();   // a right side's mark is when it starts
    f->saved_disable = false;
    fiber_install(f);
    fiber_switching = false;
    fiber_drain();
    // With SPORK_IGNORE_HB_WORK the job's clock starts here, after the switch
    // onto its fiber; beats are not yet acted on (this thread is idle).
    if constexpr (IGNORE_HB_WORK) task_mark = monotonic_ns();
    start_heartbeats();

    job->thief = WorkStealingJob::worker_id();
    job->run();

    fiber_switching = true;
    // For the join, if we are second to it.  Written before either way of
    // telling the parent we are done, and read only by a parent that waited.
    job->tokens = heartbeat_tokens;
    job->child_progress = task_elapsed(job->child_end);
    if (job->handoff) {
      if (job->join_count.fetch_sub(1, std::memory_order_acq_rel) != 2) {
        // Second to the join: take over the parent's continuation on the stack
        // the parent left behind.  Heartbeats stay armed for it.
        fiber_pending_free = f;
        jump_fcontext(job->parent_ctx, RESUME_FROM_THIEF);
        __builtin_unreachable();
      }
      // First to the join: the parent has not arrived and will continue itself.
    } else {
      job->done.test_and_set(std::memory_order_release);
    }
    fiber_pending_free = f;
    jump_fcontext(fiber_home_ctx, nullptr);
    __builtin_unreachable();
  }

  // Run one job on a fiber of its own.  Returns once some fiber has handed
  // control back to this thread.
  inline void run_job(WorkStealingJob* job) {
    spork_thread_init();
    // The caller's own computation, a pinned stack waiting at a join or a
    // worker's idle loop, is suspended for the duration.
    SporkSlot* caller_front = spork_deque_front_p;
    auto* caller_back = spork_deque_back.load();
    unsigned int caller_tokens = heartbeat_tokens;
    std::uint64_t caller_mark = task_mark;
    bool caller_disable = disable_heartbeats;
    Fiber* caller_fiber = current_fiber;

    Fiber* f = fiber_acquire();
    f->job = job;
    fiber_prepare(f);
    fiber_switching = true;
    transfer_t t = jump_fcontext(f->entry, f);
    // A fiber handed control back.  If a parent left a join for us to settle,
    // settle it here, off that parent's stack: only once the counter says the
    // parent was first is its stack free for the thief to jump onto.
    while (fiber_post_join != nullptr) {
      WorkStealingJob* j = fiber_post_join;
      fiber_post_join = nullptr;
      // t.fctx is that parent's continuation, which exists only because the
      // parent has already vacated its stack.  Publish it before the counter
      // makes the stack available to the thief.
      j->parent_ctx = t.fctx;
      if (j->join_count.fetch_sub(1, std::memory_order_acq_rel) == 2) break;  // the parent was first
      t = jump_fcontext(j->parent_ctx, RESUME_FROM_HOME);   // the thief beat us; hand it back
    }
    pause_heartbeats();
    fiber_drain();
    current_fiber = caller_fiber;
    spork_deque_front_p = caller_front;
    spork_deque_back = caller_back;
    heartbeat_tokens = caller_tokens;
    task_mark = caller_mark;
    disable_heartbeats = caller_disable;
    fiber_switching = false;
  }

  // A job nobody stole is run here (fast_clone); at the join of a stolen one
  // the continuation's tokens and mark come from both sides (join_tokens).
  inline void WorkStealingJob::sync() {
    // With SPORK_IGNORE_HB_WORK the parent's clock stops here, on arrival.
    std::uint64_t arrived = 0;
    if constexpr (IGNORE_HB_WORK) arrived = monotonic_ns();
#ifdef SPORK_HEARTBEAT_TIMING
    HeartbeatTiming* const timing = heartbeat_timing_mine;
    if (timing) timing_add(timing->joins, 1);
#endif
    if (try_dequeue()) {              // never stolen: run it here
      fast_clone(arrived);
      return;
    }
#ifdef SPORK_HEARTBEAT_TIMING
    if (timing) timing_add(timing->stolen_joins, 1);
#endif
    if (!handoff) {                   // parent's stack is pinned: wait for the child
      if (!finished()) {              // the child will be second
        pause_heartbeats();           // before reading our clock and balance: no beat may change them now
        const std::uint64_t my_progress = IGNORE_HB_WORK ? clock_at(arrived) : task_elapsed();
        const unsigned int my_tokens = heartbeat_tokens;
        get_current_scheduler().wait_for_job(this);
        join_tokens(my_progress, my_tokens, child_progress, tokens);
        start_heartbeats();
        return;
      }
      join_tokens_running(child_progress, tokens, arrived);
      return;
    }
    if (join_count.load(std::memory_order_acquire) != 1) {
      // The thief has not finished, so we may be first.  Capture this
      // continuation and get off the stack before saying so: the moment the
      // counter reaches one, the thief may jump onto this stack.
      Fiber* me = current_fiber;
      fiber_switching = true;
      fiber_save(me);
      parent_progress = IGNORE_HB_WORK ? clock_at(arrived) : task_elapsed();
      fiber_post_join = this;
      transfer_t t = jump_fcontext(fiber_home_ctx, nullptr);
      // Resumed.  A thief resuming us runs on a thread whose home handle its
      // own fiber already installed; our own idle loop hands us a fresh one.
      if (t.data == RESUME_FROM_HOME) fiber_home_ctx = t.fctx;
      // Resumed: by the thief if it arrived second, or by our own thread's
      // idle loop if the counter showed we were second after all.
      fiber_install(me);   // our balance is back in heartbeat_tokens
      if (t.data == RESUME_FROM_THIEF)   // the thief was second: its reading is the join's time
        join_tokens(parent_progress, heartbeat_tokens, child_progress, tokens, child_end);
      else                               // we were second after all
        join_tokens(child_progress, tokens, parent_progress, heartbeat_tokens);
      fiber_switching = false;
      fiber_drain();
      return;
    }
    // Otherwise the thief had already finished, so we are trivially second
    // and never left this stack.
    join_tokens_running(child_progress, tokens, arrived);
  }

  PARLAY_NOINLINE inline void SporkSlot::eager_promote() {
#ifdef SPORK_HEARTBEAT_TIMING
    const std::uint64_t timing_start = __rdtsc();
#endif
    bool before = disable_heartbeats;
    disable_heartbeats = true;
    if (heartbeat_tokens && !promoted) {
      std::uint64_t start = 0;
      if constexpr (IGNORE_HB_WORK) start = monotonic_ns();
      promote();
      // With SPORK_IGNORE_HB_WORK the promotion's time is not the
      // computation's work (beats are disabled, so the mark is ours to move).
      if constexpr (IGNORE_HB_WORK) task_mark = task_mark + (monotonic_ns() - start);
    }
    disable_heartbeats = before;
#ifdef SPORK_HEARTBEAT_TIMING
    if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->eager_ticks, __rdtsc() - timing_start);
#endif
  }

  inline volatile unsigned int* num_heartbeats = nullptr;
  inline volatile unsigned int* missed_heartbeats = nullptr;

  inline void init_heartbeat_stats() {
    if (const char* e = std::getenv("SPORK_JOIN_THIEF_TRIES")) JOIN_THIEF_TRIES = std::max(0, std::atoi(e));
    if (const char* e = std::getenv("SPORK_JOIN_THIEF_DELAY_NS")) JOIN_THIEF_DELAY_NS = std::max(0, std::atoi(e));
#ifdef RECORD_HEARTBEAT_STATS
    static bool initialized = false;
    if (!initialized) {
      initialized = true;
      unsigned int nw = parlay::internal::init_num_workers();
      num_heartbeats = new unsigned int[nw];
      missed_heartbeats = new unsigned int[nw];
      for (unsigned int wi = 0; wi < nw; ++wi) {
        num_heartbeats[wi] = 0;
        missed_heartbeats[wi] = 0;
      }
    }
#endif
  }

  inline void reset_heartbeat_stats() {
#ifdef RECORD_HEARTBEAT_STATS
    unsigned int nw = WorkStealingJob::num_workers();
    for (unsigned int wi = 0; wi < nw; ++wi) {
      num_heartbeats[wi] = 0;
      missed_heartbeats[wi] = 0;
    }
#endif
  }


  inline void heartbeat_handler(int sig) {
#ifdef SPORK_HEARTBEAT_TIMING
    const std::uint64_t timing_start = __rdtsc();
#endif
    if (in_heartbeat_handler) return;   // this beat interrupted the handler (SA_NODEFER)
    if (heartbeats_done.load(std::memory_order_relaxed)) return;
    in_heartbeat_handler = true;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    int saved_errno = errno;
    const std::uint64_t now = monotonic_ns();
    // fiber_switching covers the window in which the thread-local view of the
    // computation (deque pointers, tokens) does not match the stack it is
    // about to run on.  A beat lost there is harmless; promoting from the
    // wrong stack is not.
    if (!disable_heartbeats && !heartbeats_idle && !fiber_switching) {
#ifdef RECORD_HEARTBEAT_STATS
      volatile unsigned int& hbs = num_heartbeats[spork::WorkStealingJob::worker_id()];
      hbs = hbs + 1;
#endif
      // Pro-rated: settle the clock since the mark, at TOKENS_PER_HEARTBEAT
      // tokens per interval, together with the balance.  Below the cap the
      // remainder (under one token's worth) stays on the clock, possibly
      // reaching back past this beat, and a beat that converts nothing leaves
      // the mark alone; at the cap the rest is dropped.  A clock of at least N,
      // from a late or missed beat, is worth TOKENS_PER_HEARTBEAT tokens by
      // itself, so by default it reaches the cap.
      const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
      settle(std::uint64_t{TOKENS_PER_HEARTBEAT} * (now - task_mark) + N * heartbeat_tokens, N, now);
#ifdef SPORK_HEARTBEAT_TIMING
      const std::uint64_t promotions_before = heartbeat_timing_mine ? heartbeat_timing_mine->promotions : 0;
#endif
      SporkSlot::promote_front();
#ifdef SPORK_HEARTBEAT_TIMING
      if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->handler_promotions, t->promotions - promotions_before);
#endif
      // With SPORK_IGNORE_HB_WORK, this beat's own time, the handler and the
      // promotions it made, is not the computation's work: move the mark past
      // it, so that no later conversion credits it as clock time (in the
      // proofs a beat takes no time).  The mark stays no later than now, so
      // the clock stays non-negative.  Beats that are not acted on below
      // leave the mark alone: the interrupted code may be updating it (beats
      // disabled, a fiber switch), or there is no running computation (idle).
      // The kernel's part of a beat, delivering the signal and returning from
      // it, is not seen here and stays on the clock.
      if constexpr (IGNORE_HB_WORK) task_mark = task_mark + (monotonic_ns() - now);
    } else {
#ifdef RECORD_HEARTBEAT_STATS
      volatile unsigned int& mhbs = missed_heartbeats[spork::WorkStealingJob::worker_id()];
      mhbs = mhbs + 1;
#endif
    }
#ifdef SPORK_HEARTBEAT_TIMING
    if (HeartbeatTiming* t = heartbeat_timing_mine) {
      const std::uint64_t d = __rdtsc() - timing_start;
      if (heartbeats_idle) {
        timing_add(t->beats_idle, 1);
        timing_add(t->handler_ticks_idle, d);
      } else {
        timing_add(t->beats_busy, 1);
        timing_add(t->handler_ticks_busy, d);
      }
    }
#endif
    errno = saved_errno;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    in_heartbeat_handler = false;
  }

  inline constinit thread_local timer_t heartbeat_timer;
  inline constinit thread_local bool heartbeat_timer_live = false;
  inline constinit itimerspec heartbeat_its_zero = {};

  consteval itimerspec init_heartbeat_its() {
    itimerspec its = {};
    its.it_value   .tv_nsec = HEARTBEAT_INTERVAL_US * 1000;
    its.it_interval.tv_nsec = HEARTBEAT_INTERVAL_US * 1000;
    return its;
  }
  inline constinit itimerspec heartbeat_its = init_heartbeat_its();


inline void start_heartbeats() noexcept {
  constinit static thread_local bool thread_initialized = false;
  if (!thread_initialized) { // only first time
    thread_initialized = true;
    spork_thread_init();

    struct sigaction sa = {};
    sa.sa_flags |= SA_RESTART | SA_NODEFER;   // see in_heartbeat_handler
    sa.sa_handler = heartbeat_handler;
    sigaction(SIGALRM, &sa, nullptr);

    struct sigevent sev{};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo  = SIGALRM;
    sev._sigev_un._tid = gettid();

    timer_create(CLOCK_MONOTONIC, &sev, &heartbeat_timer);
    heartbeat_timer_live = true;
    if (current_fiber == nullptr) task_mark = monotonic_ns();   // a thread's own stack starts its mark here
#ifdef SPORK_HEARTBEAT_TIMING
    heartbeat_timing_register();
#endif
    // Each thread's beats come at its own phase of the interval: worker k of
    // P at k/P of the way through it, on CLOCK_MONOTONIC's grid of intervals,
    // so that the P threads' beats are spread evenly rather than arriving
    // together.  Every signal takes the process-wide signal lock several
    // times, and beats that arrive together queue on it: on the Xeon 8380
    // with 80 threads, 74.6 us a signal with every timer in phase, 1.9 us
    // spread evenly (as with one thread).  Armed relative to now instead, a
    // thread's phase is wherever it started its first job, and the workers
    // all start their first jobs at about the same time.
    const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
    const std::uint64_t phase = (N * WorkStealingJob::worker_id() / WorkStealingJob::num_workers()) % N;
    const std::uint64_t now = monotonic_ns();
    std::uint64_t first = now - now % N + phase;
    if (first <= now) first += N;
    itimerspec its = heartbeat_its;   // the interval
    its.it_value.tv_sec = static_cast<time_t>(first / 1000000000u);
    its.it_value.tv_nsec = static_cast<long>(first % 1000000000u);
    timer_settime(heartbeat_timer, TIMER_ABSTIME, &its, nullptr);
  }

#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) {
    if (const std::uint64_t since = t->idle_since) {
      timing_add(t->idle_ticks, __rdtsc() - since);
      __atomic_store_n(&t->idle_since, std::uint64_t{0}, __ATOMIC_RELAXED);
    }
  }
#endif
  heartbeats_idle = false;
}

// Disarm for good: called when a worker leaves the scheduler, and by the main
// thread when the scheduler shuts down.
inline void stop_heartbeats() noexcept {
  heartbeats_idle = true;
  if (heartbeat_timer_live) {
    timer_settime(heartbeat_timer, 0, &heartbeat_its_zero, nullptr);
    timer_delete(heartbeat_timer);
    heartbeat_timer_live = false;
  }
}

inline void pause_heartbeats() noexcept {
  heartbeats_idle = true;
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine)
    if (t->idle_since == 0) __atomic_store_n(&t->idle_since, __rdtsc(), __ATOMIC_RELAXED);
#endif
}

// Run body() as the left side of a promotable fork whose slot is `slot`
// (built, not yet linked; it is linked here, and promoted at once if this
// computation holds tokens).  Returns whether it was promoted.
template <typename BodyLambda>
PARLAY_INLINE bool spork_slot(SporkSlot& slot, const BodyLambda& body) {
  static_assert(std::is_invocable_v<const BodyLambda&>);
  slot.link();
  if (heartbeat_tokens) [[unlikely]]
    slot.eager_promote();
  body();
  return slot.close();
}

// The general form: prom() is the promotion.
template <typename BodyLambda, typename PromLambda>
PARLAY_INLINE bool spork(const BodyLambda& body, const PromLambda& prom) {
  static_assert(std::is_invocable_v<const BodyLambda&>);
  static_assert(std::is_invocable_v<const PromLambda&>);

  struct Slot : SporkSlot {
    const PromLambda& prom;
    explicit Slot(const PromLambda& p) noexcept
      : SporkSlot([](SporkSlot* s) { static_cast<Slot*>(s)->prom(); }), prom(p) {}
  } slot(prom);
  return spork_slot(slot, body);
}

} // namespace spork

namespace parlay {
  using WorkStealingJob = spork::WorkStealingJob;
}

#endif // PARLAY_INTERNAL_SPORK_SCHEDULER_H_
