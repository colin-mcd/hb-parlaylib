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

#define VOLATILE_UNROLL

#ifdef VOLATILE_UNROLL
extern "C" void __spork_unroll_loop(const void* site) noexcept;
extern "C" unsigned int __spork_get_unroll_factor(const void* site) noexcept;
#endif

namespace spork {

template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> seqfor(idx i, idx j, const BodyLambda&& body, const BinOp&& binop) {
  static_assert(parlay::is_monoid_v<BinOp>);
  using A = parlay::monoid_value_type_t<BinOp>;
  static_assert(std::is_invocable_r_v<A, BodyLambda&, idx>);

  if (i >= j) return fwd(binop).identity;
  A a = fwd(body)(i);
  for (++i; i < j; i++) { a = fwd(binop)(std::move(a), fwd(body)(i)); }
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

  // Runs body(k) for k in [i, j).
  //
  // For a body that returns a value, the results are folded with binop into
  // `a`, which the caller has already seeded with body of an earlier element
  // (a promoted piece seeds itself from its own first element before entering
  // here).  The monoid's identity is therefore never combined with an element,
  // which parlaylib's reduce also guarantees and callers such as min_element
  // rely on: their "identity" is a placeholder index one past the end.
  //
  // For a void body there is no accumulator and nothing is stored per
  // iteration.  That matters: a promoted piece's accumulator lives in the
  // victim's frame, next to the progress variables the victim's loop touches
  // on every iteration, so a per-iteration store from a thief would bounce
  // that cache line between the two cores for the whole loop.
  template <typename idx, typename BodyLambda, typename BinOp>
  void parfor_(idx i, idx j, parlay::monoid_value_type_t<BinOp>& a, const BodyLambda&& body, const BinOp&& binop) {
    static_assert(std::is_integral_v<idx>);
    static_assert(parlay::is_monoid_v<BinOp>);
    using A = parlay::monoid_value_type_t<BinOp>;
    using R = std::invoke_result_t<BodyLambda&, idx>;
    constexpr bool kVoid = std::is_void_v<R>;
    static_assert(kVoid || std::is_convertible_v<R, A>);
    static_assert(sizeof(sig_atomic_t) >= sizeof(idx));

    struct SpwnJob : WorkStealingJob {
      volatile idx i, j;
      const BodyLambda&& body;
      const BinOp&& binop;
      A a;
      void run() override {
        idx first = i;
        if (first >= j) { a = fwd(binop).identity; return; }   // never combined: an empty piece is not synced
        if constexpr (kVoid) {
          parfor_<idx, BodyLambda, BinOp>(first, j, a, fwd(body), fwd(binop));
        } else {
          a = fwd(body)(first);
          parfor_<idx, BodyLambda, BinOp>(first + 1, j, a, fwd(body), fwd(binop));
        }
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

    bool promoted = spork(
      [&, body = fwd(body)] () __attribute__((always_inline)) {
        #ifdef VOLATILE_UNROLL
        __spork_unroll_loop(&unroll_site);
        #endif
        for (; i < loop_end; ) {
          if constexpr (kVoid) body(i);
          else a = fwd(binop)(std::move(a), body(i));
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
        r.enqueue((heartbeat_tokens + 1) >> 1);

        if (prom_i >= mid) { l.i = 0; l.j = 0; return; }
        l.i = prom_i;
        l.j = mid;
        l.enqueue(heartbeat_tokens);
      });
    if (promoted) [[unlikely]] {
      if (l.i < l.j) [[likely]] {
        l.sync(true);
        if constexpr (!kVoid) a = fwd(binop)(a, l.a);
      }
      if (r.i < r.j) [[likely]] {
        r.sync(true); // was false
        if constexpr (!kVoid) a = fwd(binop)(a, r.a);
      }
    }
  }
} // private

// Runs body(k) for k in [i, j), folding value-returning bodies into `a` (see
// parfor_ for the seeding contract); `a` is untouched for an empty range.
template <typename idx, typename BodyLambda, typename BinOp>
void parfor(idx i, idx j, parlay::monoid_value_type_t<BinOp>& a, const BodyLambda&& body, const BinOp&& binop) {
  using A = parlay::monoid_value_type_t<BinOp>;
  using R = std::invoke_result_t<BodyLambda&, idx>;
  if constexpr (sizeof(sig_atomic_t) < sizeof(idx)) {
    if (i >= j) return;

    static_assert(sizeof(sig_atomic_t) >= sizeof(uint32_t));

    if (std::is_signed_v<idx> &&
        i >= SIG_ATOMIC_MIN &&
        j <= SIG_ATOMIC_MAX) {
      parfor_(static_cast<sig_atomic_t>(i), static_cast<sig_atomic_t>(j), a,
              [body = fwd(body)] (sig_atomic_t k) -> R { return body(static_cast<idx>(k)); },
              fwd(binop));
    } else if (std::is_unsigned_v<idx> &&
               sizeof(sig_atomic_t) <= sizeof(uint32_t) &&
               j <= 0xFFFFFFFFU) {
      parfor_(static_cast<uint32_t>(i), static_cast<uint32_t>(j), a,
              [body = fwd(body)] (uint32_t k) -> R { return body(static_cast<idx>(k)); },
              fwd(binop));
    } else if ((j - i) <= 0xFFFFFFFFU) {
      parfor_((uint32_t) 0, static_cast<uint32_t>(j - i), a,
              [i, body = fwd(body)] (uint32_t k) -> R { return body(i + static_cast<idx>(k)); },
              fwd(binop));
    } else {
      // Ranges beyond 32 bits go in blocks of 2^30; a block is one element of
      // an outer parfor and is folded like any other piece.
      const idx n = j - i;
      const idx num_blocks = 1 + ((n - 1) >> 30);
      const std::function<R(idx)> block = [n, i, &body, &binop] (idx b) -> R {
        const idx offset = b << 30;
        const idx base = i + offset;
        const uint32_t block_end = static_cast<uint32_t>(std::min<idx>(idx{1} << 30, n - offset));
        auto inner = [base, &body] (uint32_t k) -> R { return fwd(body)(base + static_cast<idx>(k)); };
        if constexpr (std::is_void_v<R>) {
          A acc = fwd(binop).identity;   // never written by a void body
          parfor_((uint32_t) 0, block_end, acc, std::move(inner), fwd(binop));
        } else {
          A acc = fwd(body)(base);
          parfor_((uint32_t) 1, block_end, acc, std::move(inner), fwd(binop));
          return acc;
        }
      };
      parfor((idx) 0, num_blocks, a, fwd(block), fwd(binop));
    }
  } else {
    parfor_(i, j, a, fwd(body), fwd(binop));
  }
}

// The fold of body(k) over [i, j); the identity only for an empty range.
template <typename idx, typename BodyLambda, typename BinOp>
parlay::monoid_value_type_t<BinOp> parfor(idx i, idx j, const BodyLambda&& body, const BinOp&& binop) {
  static_assert(!std::is_void_v<std::invoke_result_t<BodyLambda&, idx>>);
  if (i >= j) return fwd(binop).identity;
  parlay::monoid_value_type_t<BinOp> a = fwd(body)(i);
  parfor(static_cast<idx>(i + 1), j, a, fwd(body), fwd(binop));
  return a;
}

// Plain parallel loop: no accumulator, nothing stored per iteration.
template <typename idx, typename BodyLambda>
void parfor(idx i, idx j, const BodyLambda&& body) {
  char _ = 0;   // the accumulator of a void body is never written
  parfor(i, j, _, [body = fwd(body)] (idx i) { body(i); }, parlay::plus<char>());
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
