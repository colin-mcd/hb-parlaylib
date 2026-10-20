#ifndef PARLAY_INTERNAL_SPORK_PAR_H_
#define PARLAY_INTERNAL_SPORK_PAR_H_

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

  if (spork_slot(slot, lamL)) [[unlikely]] { // promoted
    slot.storage.job.sync();
  } else [[likely]] { // unpromoted
    lamR();
  }
}

} // namespace spork

#endif // PARLAY_INTERNAL_SPORK_PAR_H_
