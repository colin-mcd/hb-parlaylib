#ifndef PARLAY_INTERNAL_SPORK_DC_PARFOR_H_
#define PARLAY_INTERNAL_SPORK_DC_PARFOR_H_

// Spork's parallel loops as divide-and-conquer recursion, used in place of
// parfor.h's loop slots when built with -DSPORK_DC_LOOPS.  parfor_
// below splits [i, j) at its midpoint into a fork: the left half runs here,
// by recursion, while the right half is the fork's promotable side.  Ranges
// of at most DC_GRAIN iterations (-DSPORK_DC_GRAIN=G, default 1) are not
// split, and run as plain loops: no progress word, no volatile bound, and
// nothing for the unroll plugin to do.
//
// A fork's right half continues the left half's fold whenever it runs on
// this stack.  Unpromoted, or promoted but found unstolen at the join, it
// starts from the accumulator the left half returned rather than from
// binop.identity, and no binop combines the halves.  Only a stolen right half
// starts from the identity, on its thief, and only its result is combined,
// with binop, at the join.  So the right half is the fork's tail.  parfor_
// runs it as the next iteration of its loop, which splits it in turn, rather
// than as a tail call, which LLVM would not make: the fork's slot is in this
// frame, and its address is in the prom handler deque for the duration of
// the left half, so no later call in the frame may reuse it.
//
// A promoted right half found unstolen is dropped, as abandon drops it, with
// its tokens going back to this computation as at the join of a job that took
// no time; it then runs here as if never promoted, on this computation's
// clock and with its balance.  (A promoted fork in par instead runs its
// unstolen job on a fresh clock with the job's own tokens, then pools the two
// sides at the join: fast_clone.)  Under MPL's token scheme
// (-DSPORK_USE_MPL_HB, tokens.h), its tokens are dropped with it.
//
// Spork's parallel loops call their body PARLAY_INLINE_CALL, as in
// parfor.h, which says why.

#include "scheduler.h"
#include "../capped_copy.h"
#include "../../monoid.h"
#include "../../portability.h"

#include <cstddef>
#include <exception>
#include <new>
#include <type_traits>
#include <utility>

namespace spork {

// Ranges of at most DC_GRAIN iterations are never split (-DSPORK_DC_GRAIN=G).
#ifdef SPORK_DC_GRAIN
inline constexpr std::size_t DC_GRAIN = SPORK_DC_GRAIN;
#else
inline constexpr std::size_t DC_GRAIN = 1;
#endif
static_assert(DC_GRAIN >= 1, "SPORK_DC_GRAIN must be at least 1");

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> seqfor(idx i, idx j, const BodyLambda& body, const BinOp& binop) {
  static_assert(parlay::is_monoid_v<BinOp>);
  using A = parlay::monoid_value_type_t<BinOp>;
  static_assert(std::is_invocable_r_v<void, const BodyLambda&, idx, A&>);

  A a = binop.identity;
  for (; i < j; i++) { body(i, a); }
  return a;
}

template <typename idx, typename BodyLambda>
void seqfor(idx i, idx j, const BodyLambda& body) {
  static_assert(std::is_invocable_r_v<void, const BodyLambda&, idx>);
  for (; i < j; i++) body(i);
}

template <typename idx, typename BodyLambda, typename BinOp>
  requires parlay::is_monoid_v<BinOp>
parlay::monoid_value_type_t<BinOp> seqfor(idx n, const BodyLambda& body, const BinOp& binop) {
  return seqfor((idx) 0, n, body, binop);
}

template <typename idx, typename BodyLambda>
void seqfor(idx n, const BodyLambda& body) {
  seqfor((idx) 0, n, body);
}

namespace { // private
  template <typename idx>
  PARLAY_INLINE constexpr const idx midpoint(idx i, idx j) noexcept {
    static_assert(std::is_integral_v<idx>);
    return i + ((j - i) / 2);
  }

  // Folds body over [i, j) into a, splitting it as above.  Never inlined:
  // it recurses, and its jobs call it too.
  template <typename idx, typename BodyLambda, typename BinOp>
  PARLAY_NOINLINE parlay::monoid_value_type_t<BinOp> parfor_(idx i, idx j, parlay::monoid_value_type_t<BinOp> a, const BodyLambda& body, const BinOp& binop) {
    static_assert(std::is_integral_v<idx>);
    static_assert(parlay::is_monoid_v<BinOp>);
    using A = parlay::monoid_value_type_t<BinOp>;
    static_assert(std::is_invocable_r_v<void, const BodyLambda&, idx, A&>);

    // A promoted fork's right half, [i, j).  Its result is built by run(),
    // by whoever runs it, in raw storage: building the job (in the signal
    // handler, by the promotion) then constructs no A, so the job is plain
    // stores to build and trivial to destroy, whatever the accumulator.
    struct SpwnJob : WorkStealingJob {
      idx i, j;
      const BodyLambda& body;
      const BinOp& binop;
      alignas(A) unsigned char result[sizeof(A)];
      void run() override {
        ::new (static_cast<void*>(result)) A(parfor_<idx, BodyLambda, BinOp>(i, j, binop.identity, body, binop));
      }
      A& folded() noexcept { return *std::launder(reinterpret_cast<A*>(result)); }
      SpwnJob(idx _i, idx _j, const BodyLambda& _body, const BinOp& _binop) :
        WorkStealingJob(),
        i(_i), j(_j),
        body(_body),
        binop(_binop) {}
    };
    static_assert(std::is_trivially_destructible_v<SpwnJob>);

    // One slot for all the forks this call makes, one at a time: each offers
    // the rest of the range, [mid, end), with room for its job, which only a
    // promotion builds, as in par.  mid and end are set before each fork
    // links the slot, and read by its promotion.
    struct Slot : PromHandler {
      idx mid, end;
      const BodyLambda* body;
      const BinOp* binop;
      union Storage { Storage() {} ~Storage() {} SpwnJob job; } storage;

      Slot(const BodyLambda* f, const BinOp* op) noexcept : PromHandler(&promotion), body(f), binop(op) {}

      static void promotion(PromHandler* s) {
        Slot& slot = *static_cast<Slot*>(s);
        ::new (&slot.storage.job) SpwnJob(slot.mid, slot.end, *slot.body, *slot.binop);
        slot.storage.job.enqueue(split_tokens());
      }
    } slot(&body, &binop);
    SpwnJob& job = slot.storage.job;

    // A range too short to split, as a plain loop over a copy of the body:
    // the loop's stores cannot clobber the copy's captures.
    const auto run_leaf = [&a, &body] (idx k, idx e) PARLAY_ALWAYS_INLINE {
      const auto b = body;
      for (; k < e; ++k) PARLAY_INLINE_CALL b(k, a);
    };

    while (std::cmp_greater(j - i, DC_GRAIN)) {
      const idx mid = midpoint<idx>(i, j);
      slot.mid = mid;
      slot.end = j;
      // If the left half throws, its exception is the one that goes on, as it
      // would serially; handled as in par, whose comments say why.
      bool left_threw = false;
      union Kept { Kept() {} ~Kept() {} std::exception_ptr e; } kept;
      spork_open(slot);
      SPORK_TRY {
        if (std::cmp_less_equal(mid - i, DC_GRAIN)) run_leaf(i, mid);
        else a = parfor_<idx, BodyLambda, BinOp>(i, mid, std::move(a), body, binop);
      } SPORK_CATCH_ALL {
        ::new (&kept.e) std::exception_ptr(std::current_exception());
        left_threw = true;
      }
      if (left_threw) [[unlikely]] {
        if (slot.close()) job.abandon();
        throw_kept(kept.e);   // never returns (but see throw_kept)
        return A{};
      }
      if (slot.close()) [[unlikely]] {   // promoted: join it, unless it is unstolen
#ifdef SPORK_HEARTBEAT_TIMING
        HeartbeatTiming* const timing = heartbeat_timing_mine;
        if (timing) timing_add(timing->joins, 1);
#endif
        if (!WorkStealingJob::try_dequeue()) {   // stolen: it folded [mid, j) from the identity
#ifdef SPORK_HEARTBEAT_TIMING
          if (timing) timing_add(timing->stolen_joins, 1);
#endif
          job.join_stolen();
          if (job.threw) [[unlikely]] {
            throw_kept(job.kept_exception());   // never returns (but see throw_kept)
            return A{};
          }
          A r = std::move(job.folded());
          job.folded().~A();
          return binop(std::move(a), std::move(r));
        }
        // Unstolen: dropped, its tokens back to this computation, or under
        // MPL's scheme dropped with it (see above); HBC's scheme has none.
        if constexpr (UNSTOLEN_JOIN_TOKENS) join_tokens_running(0, job.tokens);
      }
      i = mid;   // the right half, folding on from the left half's a
    }
    run_leaf(i, j);
    return a;
  }
} // private

// Runs body(k, a) for k in [i, j), for any integral idx.  Forced inline, so
// that a caller pays a compare before the call, never a frame.
template <typename idx, typename BodyLambda, typename BinOp>
PARLAY_INLINE parlay::monoid_value_type_t<BinOp> parfor(idx i, idx j, parlay::monoid_value_type_t<BinOp> a, const BodyLambda& body, const BinOp& binop) {
  if (i >= j) return a;
  return parfor_(i, j, a, body, binop);
}

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> parfor(idx i, idx j, const BodyLambda& body, const BinOp& binop) {
  return parfor(i, j, binop.identity, body, binop);
}

template <typename idx, typename BodyLambda>
void parfor(idx i, idx j, const BodyLambda& body) {
  char _ = parfor(i, j, [body = body] (idx i, char _) { PARLAY_INLINE_CALL body(i); }, parlay::plus<char>());
}

template <typename idx, typename BodyLambda>
void parfor(idx n, const BodyLambda& body) {
  parfor((idx) 0, n, body);
}

template <typename idx, typename BodyLambda, typename BinOp>
  requires parlay::is_monoid_v<BinOp>
parlay::monoid_value_type_t<BinOp> parfor(idx n, const BodyLambda& body, const BinOp& binop) {
  return parfor((idx) 0, n, body, binop);
}

} // namespace spork

#endif // PARLAY_INTERNAL_SPORK_DC_PARFOR_H_
