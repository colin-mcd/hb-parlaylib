#ifndef PARLAY_INTERNAL_SPORK_PARFOR_H_
#define PARLAY_INTERNAL_SPORK_PARFOR_H_

#include "spork_scheduler.h"
#include "capped_copy.h"
#include "../monoid.h"
#include "../portability.h"

#include <algorithm>
#include <atomic>
#include <bits/types/sig_atomic_t.h>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits.h>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <cassert>

// VOLATILE_UNROLL is set on the command line (-DVOLATILE_UNROLL=) together
// with the Spork unroll plugin (-fpass-plugin=.../SporkUnroll.so), which
// replaces the two calls below: it strip-mines the promotable loop so the
// per-iteration volatile bound and progress store are paid once per block.
// Without it the library is header-only and each iteration pays them.
#ifdef VOLATILE_UNROLL
extern "C" void __spork_unroll_loop(const void* site) noexcept;
extern "C" unsigned int __spork_get_unroll_factor(const void* site) noexcept;
#endif

// Spork's parallel loops call their body PARLAY_INLINE_CALL (parfor_ below,
// and the wrappers between it and parallel_for's callable in scheduler.h), so
// that a function that recurses through its own loop body (nqueens) is split
// at the function, never at the body, whatever the inliner's size estimates.
// Left to the inliner, nqueens flipped to calling its body out of line for
// every iteration (+70%) when an unrelated change made the function smaller.
// Measured 2026-10-09 against leaving it to the inliner: PBBS at 80 cores
// geomean 0.999, delaunayRefine perhaps 3% slower, binaries 0.6% larger.
// (cap-parlaylib's sequential elision forces its parallel_for the same way.)
namespace spork {

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

  // The loop of a parallel for over [i, j), as a promotable fork.  Every
  // value in [i, j] must survive a round trip through sig_atomic_t, the
  // progress word the promotion handler reads: either idx is no wider than
  // sig_atomic_t (unsigned values wrap and come back), or parfor below checked
  // the bounds.  Never inlined: it is large (a slot with room for two jobs),
  // and its own jobs call it too, so inlining it could only ever pay at one
  // call site, and whether it happened would depend on the inliner's size
  // estimates for the body and everything around it.
  template <typename idx, typename BodyLambda, typename BinOp>
  PARLAY_NOINLINE parlay::monoid_value_type_t<BinOp> parfor_(idx i, idx j, parlay::monoid_value_type_t<BinOp> a, const BodyLambda& body, const BinOp& binop) {
    static_assert(std::is_integral_v<idx>);
    static_assert(parlay::is_monoid_v<BinOp>);
    using A = parlay::monoid_value_type_t<BinOp>;
    static_assert(std::is_invocable_r_v<void, const BodyLambda&, idx, A&>);
    if constexpr (sizeof(idx) > sizeof(sig_atomic_t))
      assert(std::cmp_greater_equal(i, SIG_ATOMIC_MIN) && std::cmp_less_equal(j, SIG_ATOMIC_MAX));

    #ifdef VOLATILE_UNROLL
    static char unroll_site;
    const idx unroll_factor = __spork_get_unroll_factor(&unroll_site);
    #else
    const idx unroll_factor = 1;
    #endif

    // A range of at most unroll_factor iterations can never be split: the
    // promotion handler below declines whenever that few remain.  Run it as a
    // plain loop, without the jobs, the promotion slot or the progress
    // protocol, which cost more than a short loop's iterations do.
    if (j - i <= unroll_factor) {
      const auto b = body;   // a copy, as the loop below takes: stores cannot clobber its captures
      for (; i < j; ++i) PARLAY_INLINE_CALL b(i, a);
      return a;
    }

    struct SpwnJob : WorkStealingJob {
      volatile idx i, j;
      const BodyLambda& body;
      const BinOp& binop;
      A a;
      void run() override {
        a = parfor_<idx, BodyLambda, BinOp>(i, j, binop.identity, body, binop);
      }
      SpwnJob(const BodyLambda& _body, const BinOp& _binop) :
        WorkStealingJob(),
        body(_body),
        binop(_binop) {}
    };

    // Most loops are never promoted, so the two jobs are built by the
    // promotion handler, in storage reserved in the slot, rather than on every
    // call.  They live in this frame either way, until the syncs below.
    // Building a job default-constructs its accumulator, which runs in the
    // signal handler; an accumulator that is not trivial to construct (one
    // that may allocate or look up the scheduler) is built up front instead.
    constexpr bool lazy_jobs = std::is_trivially_default_constructible_v<A> &&
                               std::is_trivially_destructible_v<A>;
    union JobStorage { JobStorage() {} ~JobStorage() {} SpwnJob job; };

    // Main code writes progress; the signal handler only reads it.
    volatile sig_atomic_t sig_safe_i = static_cast<sig_atomic_t>(i);
    // Main code reads the bound; the signal handler may shorten it.
    volatile idx loop_end = j;

    // The loop's slot: what its promotion needs, reached through the slot
    // rather than through a closure of references built on every call (as in
    // par).  The progress word and the bound stay separate locals, pointed
    // to from here: the unroll plugin requires each to be its own stack slot.
    struct LoopSlot : SporkSlot {
      volatile sig_atomic_t* progress;
      volatile idx* bound;
      idx end;
      const BodyLambda* body;
      const BinOp* binop;
      JobStorage ls, rs;

      LoopSlot(volatile sig_atomic_t* p, volatile idx* b, idx e,
               const BodyLambda* f, const BinOp* op) noexcept
        : SporkSlot(&promotion), progress(p), bound(b), end(e), body(f), binop(op) {}

      static void promotion(SporkSlot* s) {
        LoopSlot& self = *static_cast<LoopSlot*>(s);
        SpwnJob& l = self.ls.job;
        SpwnJob& r = self.rs.job;
        // Built before any early return: promoted implies built, and the
        // code after the loop reads every job's i and j once promoted.
        if constexpr (lazy_jobs) {
          ::new (&self.ls.job) SpwnJob(*self.body, *self.binop);
          ::new (&self.rs.job) SpwnJob(*self.body, *self.binop);
        }
        #ifdef VOLATILE_UNROLL
        const idx inc_i = __spork_get_unroll_factor(&unroll_site);
        #else
        const idx inc_i = 1;
        #endif
        const idx ssi = static_cast<idx>(*self.progress);
        const idx bound = *self.bound;
        if (bound - ssi <= inc_i) { r.i = 0; r.j = 0; l.i = 0; l.j = 0; return; }
        const idx prom_i = ssi + inc_i;
        const idx mid = midpoint<idx>(prom_i, bound);
        *self.bound = prom_i;

        // Of the t tokens left after the promotion's, r gets floor(t/2) and
        // l the other ceil(t/2): l is the bulk of the left half, and this
        // loop has at most inc_i iterations to go, so it keeps none.  Without
        // an l, the left half is this loop, which keeps ceil(t/2).
        r.i = mid;
        r.j = self.end;
        r.enqueue(split_tokens());

        if (prom_i >= mid) { l.i = 0; l.j = 0; return; }
        l.i = prom_i;
        l.j = mid;
        const unsigned int left_tokens = heartbeat_tokens;
        heartbeat_tokens = 0;
        l.enqueue(left_tokens);
      }
    } slot(&sig_safe_i, &loop_end, j, &body, &binop);
    if constexpr (!lazy_jobs) {
      ::new (&slot.ls.job) SpwnJob(body, binop);
      ::new (&slot.rs.job) SpwnJob(body, binop);
    }
    SpwnJob& l = slot.ls.job;
    SpwnJob& r = slot.rs.job;

    bool promoted = false;
    // Whichever jobs were built are destroyed on the way out, by exception
    // too; a no-op for trivial ones.
    struct DestroyJobs {
      SpwnJob& l;
      SpwnJob& r;
      const bool& promoted;
      ~DestroyJobs() {
        if (!lazy_jobs || promoted) {
          std::destroy_at(&l);
          std::destroy_at(&r);
        }
      }
    } destroy_jobs{l, r, promoted};

    // Serially, an exception from this loop's own iterations comes before
    // l's, and l's before r's: the first one is thrown on once the jobs are
    // joined, and any after it are dropped (as par does, which see).
    // Handled as in par: the handlers only keep the exception, and throw_kept
    // throws it on outside them.
    bool threw = false;
    union Kept { Kept() {} ~Kept() {} std::exception_ptr e; } kept;
    const auto loop = [&, body = body] () PARLAY_ALWAYS_INLINE {
        #ifdef VOLATILE_UNROLL
        __spork_unroll_loop(&unroll_site);
        #endif
        for (; i < loop_end; ) {
          PARLAY_INLINE_CALL body(i, a);
          ++i;
          sig_safe_i = static_cast<sig_atomic_t>(i);
        }
      };
    spork_open(slot);
    SPORK_TRY {
      loop();
    } SPORK_CATCH_ALL {
      ::new (&kept.e) std::exception_ptr(std::current_exception());
      threw = true;
    }
    if (threw) [[unlikely]] {
      promoted = slot.close();
      if (promoted) {
        // Last enqueued, first joined: l, then r.
        if (l.i < l.j) l.abandon();
        if (r.i < r.j) r.abandon();
      }
      throw_kept(kept.e);   // never returns (but see throw_kept)
      return A{};
    } else if ((promoted = slot.close())) [[unlikely]] {
      // No handler here: l.join() reports an exception rather than throwing
      // it, and both jobs are joined before either result is combined, so a
      // binop that throws leaves no job behind.  A handler on this path made
      // LLVM keep a in memory in the loop above (classify's entropy
      // reduction, a double: 3.7% slower at one core).
      // (The ranges are read where they are tested, not kept across the
      // joins in registers, which would have to be callee-saved: three more
      // pushes and pops on every call.)
      if (l.i < l.j && l.join()) [[unlikely]] {
        if (r.i < r.j) r.abandon();
        throw_kept(l.kept_exception());   // never returns (but see throw_kept)
        return A{};
      }
      if (r.i < r.j) [[likely]] r.sync();
      if (l.i < l.j) [[likely]] a = binop(a, l.a);
      if (r.i < r.j) [[likely]] a = binop(a, r.a);
    }
    return a;
  }

  // A range whose bounds do not fit sig_atomic_t: blocks of at most
  // SIG_ATOMIC_MAX iterations, each a full parallel region folding into the
  // same accumulator in order.  Out of line and cold, so that the common path
  // in parfor below costs its callers nothing for it.
  template <typename idx, typename BodyLambda, typename BinOp>
  PARLAY_NOINLINE PARLAY_COLD parlay::monoid_value_type_t<BinOp> parfor_blocks(idx i, idx j, parlay::monoid_value_type_t<BinOp> a, const BodyLambda& body, const BinOp& binop) {
    using A = parlay::monoid_value_type_t<BinOp>;
    for (idx s = i; s < j; ) {
      const sig_atomic_t n = static_cast<sig_atomic_t>(std::min<idx>(SIG_ATOMIC_MAX, j - s));
      a = parfor_(sig_atomic_t{0}, n, a,
                  [s, body = body] (sig_atomic_t k, A& a) {
                    PARLAY_INLINE_CALL body(s + static_cast<idx>(k), a);
                  }, binop);
      s += n;
    }
    return a;
  }
} // private

// Runs body(k, a) for k in [i, j), for any integral idx.  A range whose
// bounds fit sig_atomic_t (parfor_'s progress word) goes straight to parfor_
// with idx itself, whatever its width, so the body gets its own index type
// with no conversion and no wrapper: that is every range of a narrow idx, and
// nearly every range of a wide one (parallel_for's size_t).  Only a range
// that does not fit runs as blocks (parfor_blocks).  Forced inline, so that
// the dispatch costs a caller a compare or two, never a call or a frame.
// (An earlier extent check with an out-of-line call for huge ranges, which
// still sent the common case through a sig_atomic_t wrapper, measured
// slower on classify: +4.7% against df87c0b, +1.3% then.)
template <typename idx, typename BodyLambda, typename BinOp>
PARLAY_INLINE parlay::monoid_value_type_t<BinOp> parfor(idx i, idx j, parlay::monoid_value_type_t<BinOp> a, const BodyLambda& body, const BinOp& binop) {
  if (i >= j) return a;
  if constexpr (sizeof(idx) <= sizeof(sig_atomic_t)) {
    return parfor_(i, j, a, body, binop);
  } else {
    if (std::cmp_greater_equal(i, SIG_ATOMIC_MIN) && std::cmp_less_equal(j, SIG_ATOMIC_MAX)) [[likely]]
      return parfor_(i, j, a, body, binop);
    return parfor_blocks(i, j, a, body, binop);
  }
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

#endif // PARLAY_INTERNAL_SPORK_PARFOR_H_
