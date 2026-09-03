#ifndef PARLAY_INTERNAL_SPORK_PARFOR_H_
#define PARLAY_INTERNAL_SPORK_PARFOR_H_

#include "spork_scheduler.h"
#include "../monoid.h"

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
        r.enqueue((heartbeat_tokens + 1) >> 1);

        if (prom_i >= mid) { l.i = 0; l.j = 0; return; }
        l.i = prom_i;
        l.j = mid;
        l.enqueue(heartbeat_tokens);
      });
    if (promoted) [[unlikely]] {
      if (l.i < l.j) [[likely]] {
        l.sync(true);
        a = fwd(binop)(a, l.a);
      }
      if (r.i < r.j) [[likely]] {
        r.sync(false);
        a = fwd(binop)(a, r.a);
      }
    }
  }
} // private

template <typename idx, typename BodyLambda, typename BinOp>
void parfor(idx i, idx j, parlay::monoid_value_type_t<BinOp>& a, const BodyLambda&& body, const BinOp&& binop) {
  using A = parlay::monoid_value_type_t<BinOp>;
  if constexpr (sizeof(sig_atomic_t) < sizeof(idx)) {
    if (i >= j) return;

    static_assert(sizeof(sig_atomic_t) >= sizeof(uint32_t));

    if (std::is_signed_v<idx> &&
        i >= SIG_ATOMIC_MIN &&
        j <= SIG_ATOMIC_MAX) {
      parfor_(static_cast<sig_atomic_t>(i), static_cast<sig_atomic_t>(j), a,
              [body = fwd(body)] (sig_atomic_t k, A& a) {
                return body(static_cast<idx>(k), a);
              }, fwd(binop));
    } else if (std::is_unsigned_v<idx> &&
               sizeof(sig_atomic_t) <= sizeof(uint32_t) &&
               j <= 0xFFFFFFFFU) {
        parfor_(static_cast<uint32_t>(i), static_cast<uint32_t>(j), a,
                [body = fwd(body)] (uint32_t k, A& a) {
                  return body(static_cast<idx>(k), a);
                }, fwd(binop));
    } else if ((j - i) <= 0xFFFFFFFFU) {
      parfor_((uint32_t) 0, static_cast<uint32_t>(j - i), a,
              [i, body = fwd(body)] (uint32_t k, A& a) {
                return body(i + static_cast<idx>(k), a);
              }, fwd(binop));
    } else {
      idx n = j - i;
      idx num_blocks = 1 + ((j - i - 1) >> 30);
      const std::function<void(idx, A&)> fn = [j, i, &body, &binop] (idx block, A& a) {
        idx offset = block << 30;
        uint32_t block_end = (block == 1 + ((j - i - 1) >> 30)) ?
          ((block + 1) << 30) : (j - i - offset);
        parfor_((uint32_t) 0, block_end, a,
                [base = i + offset, &body] (uint32_t k, A& a) {
                  fwd(body)(base + static_cast<idx>(k), a);
                }, fwd(binop));
      };
      parfor((idx) 0, num_blocks, a, fwd(fn), fwd(binop));
    }
  } else {
    parfor_(i, j, a, fwd(body), fwd(binop));
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
