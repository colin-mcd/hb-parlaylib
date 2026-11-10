#ifndef PARLAY_INTERNAL_SPORK_TIMING_H_
#define PARLAY_INTERNAL_SPORK_TIMING_H_

// What heartbeats cost, counted per thread with -DSPORK_HEARTBEAT_TIMING;
// nothing without it.

#include "spork_tokens.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#ifdef SPORK_HEARTBEAT_TIMING
#include <x86intrin.h>   // __rdtsc, for the heartbeat timing counters
#endif

namespace spork {

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

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_TIMING_H_
