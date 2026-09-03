#ifndef PARLAY_INTERNAL_SPORK_PAR_H_
#define PARLAY_INTERNAL_SPORK_PAR_H_

#include "spork_scheduler.h"

namespace spork {

template <typename LambdaL, typename LambdaR>
__attribute__((always_inline))
void parSeq(LambdaL&& lamL, LambdaR&& lamR) {
  static_assert(std::is_invocable_v<LambdaL&>);
  static_assert(std::is_invocable_v<LambdaR&>);
  fwd(lamL)();
  fwd(lamR)();
}

template <typename LambdaL, typename LambdaR>
__attribute__((always_inline))
void par(const LambdaL&& lamL, const LambdaR&& lamR) {
  static_assert(std::is_invocable_v<const LambdaL&>);
  static_assert(std::is_invocable_v<const LambdaR&>);
  
  struct SpwnJob : WorkStealingJob {
    const LambdaR&& lamR;
    void run() override { fwd(lamR)(); }
    SpwnJob(const LambdaR&& lamR) :
      WorkStealingJob(),
      lamR(fwd(lamR)) {}
  };
  
  SpwnJob jp(fwd(lamR));

  bool promoted = with_prom_handler(
    fwd(lamL),
    [&jp] () { jp.enqueue(heartbeat_tokens >> 1); });

  if (promoted) [[unlikely]] { // promoted
    jp.sync(false);
  } else [[likely]] { // unpromoted
    fwd(lamR)();
  }
}

} // namespace spork

#endif // PARLAY_INTERNAL_SPORK_PAR_H_
