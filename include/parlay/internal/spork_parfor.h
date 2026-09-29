#ifndef PARLAY_INTERNAL_SPORK_PARFOR_H_
#define PARLAY_INTERNAL_SPORK_PARFOR_H_

#include "spork_scheduler.h"
#include "../monoid.h"

#include <algorithm>
#include <atomic>
#include <bits/types/sig_atomic_t.h>
#include <cstdint>
#include <functional>
#include <limits.h>
#include <type_traits>

// VOLATILE_UNROLL is set on the command line (-DVOLATILE_UNROLL=) together
// with the Spork unroll plugin (-fpass-plugin=.../SporkUnroll.so), which
// replaces the two calls below: it strip-mines the promotable loop so the
// per-iteration volatile bound and progress store are paid once per block.
// Without it the library is header-only and each iteration pays them.
#ifdef VOLATILE_UNROLL
extern "C" void __spork_unroll_loop(const void* site) noexcept;
extern "C" unsigned int __spork_get_unroll_factor(const void* site) noexcept;
#endif

namespace spork {

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> seqfor(idx i, idx j, const BodyLambda&& body, const BinOp&& binop) {
  static_assert(parlay::is_monoid_v<BinOp>);
  using A = parlay::monoid_value_type_t<BinOp>;
  static_assert(std::is_invocable_r_v<void, BodyLambda&, idx, A&>);

  A a = fwd(binop).identity;
  for (; i < j; i++) { fwd(body)(i, a); }
  return a;
}

template <typename idx, typename BodyLambda>
void seqfor(idx i, idx j, const BodyLambda&& body) {
  static_assert(std::is_invocable_r_v<void, BodyLambda&, idx>);
  for (; i < j; i++) fwd(body)(i);
}

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> seqfor(idx n, const BodyLambda&& body, const BinOp&& binop) {
  return seqfor((idx) 0, n, fwd(body), fwd(binop));
}

template <typename idx, typename BodyLambda>
void seqfor(idx n, const BodyLambda&& body) {
  seqfor((idx) 0, n, fwd(body));
}

namespace { // private
  template <typename idx>
  __attribute__((always_inline))
  constexpr const idx midpoint(idx i, idx j) noexcept {
    static_assert(std::is_integral_v<idx>);
    return i + ((j - i) / 2);
  }

  template <typename idx, typename BodyLambda, typename BinOp>
  void parfor_(idx i, idx j, parlay::monoid_value_type_t<BinOp>& a, const BodyLambda&& body, const BinOp&& binop) {
    static_assert(std::is_integral_v<idx>);
    static_assert(parlay::is_monoid_v<BinOp>);
    using A = parlay::monoid_value_type_t<BinOp>;
    static_assert(std::is_invocable_r_v<void, BodyLambda&, idx, A&>);
    static_assert(sizeof(sig_atomic_t) >= sizeof(idx));

    struct SpwnJob : WorkStealingJob {
      volatile idx i, j;
      const BodyLambda&& body;
      const BinOp&& binop;
      A a;
      void run() override {
        a = fwd(binop).identity;
        parfor_<idx, BodyLambda, BinOp>(i, j, a, fwd(body), fwd(binop));
      }
      SpwnJob(const BodyLambda&& _body, const BinOp&& _binop) :
        WorkStealingJob(),
        body(fwd(_body)),
        binop(fwd(_binop)) {}
    };

    SpwnJob l(fwd(body), fwd(binop));
    SpwnJob r(fwd(body), fwd(binop));

    // Main code writes progress; the signal handler only reads it.
    volatile sig_atomic_t sig_safe_i = i;
    // Main code reads the bound; the signal handler may shorten it.
    volatile idx loop_end = j;
    #ifdef VOLATILE_UNROLL
    static char unroll_site;
    #endif

    bool promoted = with_prom_handler(
      [&, body = fwd(body)] () __attribute__((always_inline)) {
        #ifdef VOLATILE_UNROLL
        __spork_unroll_loop(&unroll_site);
        #endif
        for (; i < loop_end; ) {
          body(i, a);
          ++i;
          sig_safe_i = static_cast<sig_atomic_t>(i);
        }
      },
      [&] () {
        #ifdef VOLATILE_UNROLL
        idx inc_i = __spork_get_unroll_factor(&unroll_site);
        #else
        idx inc_i = 1;
        #endif
        idx ssi = static_cast<idx>(sig_safe_i);
        if (loop_end - ssi <= inc_i) { r.i = 0; r.j = 0; l.i = 0; l.j = 0; return; }
        idx prom_i = ssi + inc_i;
        idx mid = midpoint<idx>(prom_i, loop_end);
        loop_end = prom_i;

        r.i = mid;
        r.j = j;
        r.enqueue();

        if (prom_i >= mid) { l.i = 0; l.j = 0; return; }
        l.i = prom_i;
        l.j = mid;
        l.enqueue();
      });
    if (promoted) [[unlikely]] {
      if (l.i < l.j) [[likely]] {
        l.sync();
        a = fwd(binop)(a, l.a);
      }
      if (r.i < r.j) [[likely]] {
        r.sync();
        a = fwd(binop)(a, r.a);
      }
    }
  }
  // Every range too wide for parfor_ runs through this wrapper: rebased to
  // [0, n) with a sig_atomic_t index, and the full-width base added back before
  // the body sees it.  The index is signed and the body is copied rather than
  // referenced because both are measurably faster in tight loops: signed
  // overflow is undefined, so the compiler may widen the counter, and a copied
  // body's captures need not be reloaded after every store the body makes
  // (wordCounts: +13% with uint32_t and a reference).
  template <typename idx, typename BodyLambda, typename BinOp>
  void parfor_rebased(idx base, sig_atomic_t n, parlay::monoid_value_type_t<BinOp>& a,
                      const BodyLambda&& body, const BinOp&& binop) {
    using A = parlay::monoid_value_type_t<BinOp>;
    parfor_(sig_atomic_t{0}, n, a,
            [base, body = fwd(body)] (sig_atomic_t k, A& a) { body(base + static_cast<idx>(k), a); },
            fwd(binop));
  }
} // private

// Runs body(k, a) for k in [i, j), for any integral idx.  An index no wider
// than sig_atomic_t goes straight to parfor_.  A wider range runs as blocks of
// at most SIG_ATOMIC_MAX iterations, each a full parallel region folding into
// the same accumulator in order.  Nearly every range is a single block, and the
// loop condition doubles as the empty and inverted range check: this shape
// measured faster than a separate empty check, extent check and out-of-line
// call for huge ranges (classify +4.7% with those against df87c0b, +1.3% here).
template <typename idx, typename BodyLambda, typename BinOp>
void parfor(idx i, idx j, parlay::monoid_value_type_t<BinOp>& a, const BodyLambda&& body, const BinOp&& binop) {
  if constexpr (sizeof(idx) <= sizeof(sig_atomic_t)) {
    if (i >= j) return;
    parfor_(i, j, a, fwd(body), fwd(binop));
  } else {
    for (idx s = i; s < j; ) {
      const idx n = std::min<idx>(SIG_ATOMIC_MAX, j - s);
      parfor_rebased<idx, BodyLambda, BinOp>(s, static_cast<sig_atomic_t>(n), a, fwd(body), fwd(binop));
      s += n;
    }
  }
}

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> parfor(idx i, idx j, const BodyLambda&& body, const BinOp&& binop) {
  parlay::monoid_value_type_t<BinOp> a = fwd(binop).identity;
  parfor(i, j, a, fwd(body), fwd(binop));
  return a;
}

template <typename idx, typename BodyLambda>
void parfor(idx i, idx j, const BodyLambda&& body) {
  char _ = parfor(i, j, [body = fwd(body)] (idx i, char _) {body(i);}, parlay::plus<char>());
}

template <typename idx, typename BodyLambda>
void parfor(idx n, const BodyLambda&& body) {
  parfor((idx) 0, n, fwd(body));
}

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> parfor(idx n, const BodyLambda&& body, const BinOp&& binop) {
  return parfor((idx) 0, n, fwd(body), fwd(binop));
}

} // namespace spork

#endif // PARLAY_INTERNAL_SPORK_PARFOR_H_
