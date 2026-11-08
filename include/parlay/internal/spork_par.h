#ifndef PARLAY_INTERNAL_SPORK_PAR_H_
#define PARLAY_INTERNAL_SPORK_PAR_H_

#include <exception>
#include <new>
#include <type_traits>

#include "spork_scheduler.h"
#include "../portability.h"

namespace spork {

template <typename LambdaL, typename LambdaR>
PARLAY_INLINE void parSeq(const LambdaL& lamL, const LambdaR& lamR) {
  static_assert(std::is_invocable_v<LambdaL&>);
  static_assert(std::is_invocable_v<LambdaR&>);
  lamL();
  lamR();
}

template <typename LambdaL, typename LambdaR>
PARLAY_INLINE void par(const LambdaL& lamL, const LambdaR& lamR) {
  static_assert(std::is_invocable_v<const LambdaL&>);
  static_assert(std::is_invocable_v<const LambdaR&>);
  
  struct SpwnJob : WorkStealingJob {
    const LambdaR& lamR;
    void run() override { lamR(); }
    SpwnJob(const LambdaR& lamR) :
      WorkStealingJob(),
      lamR(lamR) {}
  };
  static_assert(std::is_trivially_destructible_v<SpwnJob>);

  // The fork's slot, with room for the right side's job.  The job is built
  // only if the fork is promoted, by the promotion itself (in the heartbeat
  // handler, or eagerly), as parfor_ builds its jobs: most forks never are,
  // and building a job costs a vtable pointer, two flags and a reference.
  // Those are plain stores, so the handler may make them.
  struct Slot : SporkSlot {
    const LambdaR& lamR;
    union Storage { Storage() {} ~Storage() {} SpwnJob job; } storage;

    explicit Slot(const LambdaR& r) noexcept : SporkSlot(&promotion), lamR(r) {}

    static void promotion(SporkSlot* s) {
      Slot& slot = *static_cast<Slot*>(s);
      ::new (&slot.storage.job) SpwnJob(slot.lamR);
      slot.storage.job.enqueue(split_tokens());
    }
  } slot(lamR);

  // If lamL throws, its exception is the one that goes on, as it would be
  // serially.  Handling it costs the fast path nothing: the handler is a
  // landing pad, run only by the unwinder.  The handler only keeps the
  // exception, and calls nothing that can throw.
  bool left_threw = false;
  union Kept { Kept() {} ~Kept() {} std::exception_ptr e; } kept;
  spork_open(slot);
  SPORK_TRY {
    lamL();
  } SPORK_CATCH_ALL {
    ::new (&kept.e) std::exception_ptr(std::current_exception());
    left_threw = true;
  }
  if (left_threw) [[unlikely]] {
    // Promoted, lamR's job may be running elsewhere against this frame, so
    // it is joined (abandoned) before the exception goes on.  Not inside the
    // handler: a join can resume this frame on another thread, and the C++
    // runtime keeps the exceptions being handled per thread.
    if (slot.close()) slot.storage.job.abandon();
    throw_kept(kept.e);   // never returns (but see throw_kept)
  } else if (slot.close()) [[unlikely]] { // promoted
    slot.storage.job.sync();     // throws on whatever lamR threw
    return;
  }
  lamR();   // unpromoted
}

} // namespace spork

#endif // PARLAY_INTERNAL_SPORK_PAR_H_
