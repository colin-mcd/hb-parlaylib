#ifndef PARLAY_INTERNAL_SPORK_HEARTBEAT_H_
#define PARLAY_INTERNAL_SPORK_HEARTBEAT_H_

// The heartbeat: each thread's timer, and the signal handler that converts
// the running computation's credit into tokens and promotes with them.

#include "spork_api.h"
#include "spork_deque.h"
#include "spork_fiber.h"
#include "spork_job.h"
#include "spork_timing.h"
#include "spork_tokens.h"
#include "../portability.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <ctime>

#include <unistd.h>

namespace spork {

// ---- heartbeat state --------------------------------------------------------
// true => this *processor* (= thread) is between jobs; not fiber state
inline constinit thread_local volatile bool heartbeats_idle = true;
// true => this thread's heartbeat handler is running.  The handler is
// installed SA_NODEFER (start_heartbeats), so SIGALRM stays unblocked while
// it runs: the thread's signal mask then never changes on the way into the
// handler or out of it, and the kernel skips taking the process-wide signal
// lock for those two changes.  Nothing then stops the kernel from laying a
// beat's frame over the handler, or over a frame it has set up and not yet
// entered, which is why the timer is one-shot and re-armed only by the
// handler (arm_heartbeat).  A SIGALRM from anywhere else that lands in the
// handler finds this set and is dropped.
inline constinit thread_local volatile bool in_heartbeat_handler = false;

// -DRECORD_HEARTBEAT_STATS counts each worker's beats, acted on or missed, in
// num_heartbeats and missed_heartbeats.  Off by default: the counters sit side
// by side in one array, so every beat wrote a cache line shared with 15 other
// workers, which cost the handler about 110 ns a beat at 80 threads, and
// nothing reads them (SPORK_HEARTBEAT_TIMING counts beats per thread).
inline volatile unsigned int* num_heartbeats = nullptr;
inline volatile unsigned int* missed_heartbeats = nullptr;

inline void init_heartbeat_stats() {
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

// ---- the heartbeat timer ----------------------------------------------------
// One per thread, on CLOCK_MONOTONIC, signalling only that thread.  It is
// one-shot: start_heartbeats arms it, and then each beat's handler re-arms
// it as its last step.  A periodic timer is re-armed by the kernel as it
// dequeues each signal, before the frame is even built, so with the
// handler SA_NODEFER the next beat is laid over the last one whenever
// delivering a signal takes longer than the interval.  With many threads
// and short intervals it does: every signal takes the process-wide signal
// lock three times (queueing it, dequeueing it, and again after re-arming
// the timer), and on the Xeon 8380 with 80 threads at 25 us beats, 80% of
// all cycles were spent spinning on that lock.  Frames of 3.3 KB (the
// AVX-512 state), none of which had run an instruction of the handler,
// stacked 2,518 deep and overflowed an 8 MB fiber stack (fib, 80 threads,
// 10 us).  Re-armed by its handler, a timer cannot come due while its last
// beat is still being delivered, and the kernel takes the lock twice a
// beat, not three times, as it no longer re-arms the timer.
inline constinit thread_local timer_t heartbeat_timer;
inline constinit thread_local volatile bool heartbeat_timer_live = false;
// This thread's beats fall at this offset into each interval (start_heartbeats).
inline constinit thread_local std::uint64_t heartbeat_phase = 0;
inline constinit itimerspec heartbeat_its_zero = {};

// Arm this thread's timer for the first point of its grid, phase + k * N
// for whole k, at least N / 2 from now.  Normally that is the next one; a
// beat delivered more than N / 2 late skips one, so that whatever the
// signals cost, the computation runs for at least half an interval
// between beats.
inline void arm_heartbeat() noexcept {
  const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
  const std::uint64_t now = monotonic_ns();
  std::uint64_t next = now - (now - heartbeat_phase) % N + N;
  if (next - now < N / 2) next += N;
  itimerspec its = {};   // no it_interval: one-shot
  its.it_value.tv_sec = static_cast<time_t>(next / 1000000000u);
  its.it_value.tv_nsec = static_cast<long>(next % 1000000000u);
  timer_settime(heartbeat_timer, TIMER_ABSTIME, &its, nullptr);
}

// A beat's work: convert the running computation's credit into tokens,
// then promote with them.  Run by the handler, or by HeartbeatsDisabled for
// a beat that landed while beats were disabled, with no other beat acted on
// meanwhile either way, as settle requires.
inline void run_heartbeat(std::uint64_t now) noexcept {
  // Pro-rated: settle the clock since the mark, at TOKENS_PER_HEARTBEAT
  // tokens per interval, together with the balance.  Below the cap the
  // remainder (under one token's worth) stays on the clock, possibly
  // reaching back past this beat, and a beat that converts nothing leaves
  // the mark alone; at the cap the rest is dropped.  A clock of at least N,
  // from a late or missed beat, is worth TOKENS_PER_HEARTBEAT tokens by
  // itself, so by default it reaches the cap.  With SPORK_NO_CLOCK a beat
  // just brings TOKENS_PER_HEARTBEAT tokens, up to the cap.
  if constexpr (NO_CLOCK) {
    heartbeat_tokens = std::min(heartbeat_tokens + TOKENS_PER_HEARTBEAT, TOKEN_LIMIT);
  } else {
    const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
    settle(std::uint64_t{TOKENS_PER_HEARTBEAT} * (now - task_mark) + N * heartbeat_tokens, N, now);
  }
#ifdef SPORK_HEARTBEAT_TIMING
  const std::uint64_t promotions_before = heartbeat_timing_mine ? heartbeat_timing_mine->promotions : 0;
#endif
  SporkSlot::promote_front();
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->handler_promotions, t->promotions - promotions_before);
#endif
}

// A beat left pending while beats were disabled, run as they come back on
// (~HeartbeatsDisabled).  The flag is cleared while beats are still on, so
// a beat landing in between is acted on directly; one landing while this
// runs sets it again for the destructor's next check.  Dropped if this
// thread has since gone idle or begun a switch, as the handler would.
PARLAY_NOINLINE inline void run_pending_heartbeat() noexcept {
  heartbeat_pending = false;
  disable_heartbeats = true;
  if (!heartbeats_idle && !fiber_switching) run_heartbeat(NO_CLOCK ? 0 : monotonic_ns());
  disable_heartbeats = false;
  std::atomic_signal_fence(std::memory_order_seq_cst);
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
  const std::uint64_t now = NO_CLOCK ? 0 : monotonic_ns();
  // fiber_switching covers the window in which the thread-local view of the
  // computation (deque pointers, tokens) does not match the stack it is
  // about to run on.  A beat lost there is harmless; promoting from the
  // wrong stack is not.
  if (!disable_heartbeats && !heartbeats_idle && !fiber_switching) {
#ifdef RECORD_HEARTBEAT_STATS
    volatile unsigned int& hbs = num_heartbeats[spork::WorkStealingJob::worker_id()];
    hbs = hbs + 1;
#endif
    run_heartbeat(now);
  } else {
    // With beats disabled the interrupted code is reading or rewriting the
    // balance or the mark: leave the beat for HeartbeatsDisabled to run once
    // they are back on.  Idle or mid-switch there is no running computation
    // to credit, and the beat is dropped.
    if (!heartbeats_idle && !fiber_switching) heartbeat_pending = true;
#ifdef RECORD_HEARTBEAT_STATS
    volatile unsigned int& mhbs = missed_heartbeats[spork::WorkStealingJob::worker_id()];
    mhbs = mhbs + 1;
#endif
  }
  std::atomic_signal_fence(std::memory_order_seq_cst);
  in_heartbeat_handler = false;
  // The next beat can come due from here on, so nothing but bookkeeping
  // follows.  A timer stopped for good (stop_heartbeats) stays stopped.
  if (heartbeat_timer_live) arm_heartbeat();
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
}

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
    if (!NO_CLOCK && current_fiber == nullptr) task_mark = monotonic_ns();   // a thread's own stack starts its mark here
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
    // all start their first jobs at about the same time.  Each beat's
    // handler re-arms the timer on the same grid (arm_heartbeat).
    const std::uint64_t N = HEARTBEAT_INTERVAL_NS;
    heartbeat_phase = (N * WorkStealingJob::worker_id() / WorkStealingJob::num_workers()) % N;
    arm_heartbeat();
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
    heartbeat_timer_live = false;   // first: a beat landing below must not re-arm the timer
    std::atomic_signal_fence(std::memory_order_seq_cst);
    timer_settime(heartbeat_timer, 0, &heartbeat_its_zero, nullptr);
    timer_delete(heartbeat_timer);
  }
}

inline void pause_heartbeats() noexcept {
  heartbeats_idle = true;
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine)
    if (t->idle_since == 0) __atomic_store_n(&t->idle_since, __rdtsc(), __ATOMIC_RELAXED);
#endif
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_HEARTBEAT_H_
