#ifndef PARLAY_INTERNAL_CAPPED_COPY_H_
#define PARLAY_INTERNAL_CAPPED_COPY_H_

#include <cstddef>

#include <memory>
#include <type_traits>

// How spork's parallel loops call their body (parfor_, and the wrappers
// between it and parallel_for's callable): always forced inline, so that a
// function that recurses through its own loop body (nqueens) is split at the
// function, never at the body, whatever the inliner's size estimates.  Left
// to the inliner, nqueens flipped to calling its body out of line for every
// iteration (+70%) when an unrelated change made the function smaller.
// Measured 2026-10-09 against leaving it to the inliner: PBBS at 80 cores
// geomean 0.999, delaunayRefine perhaps 3% slower, binaries 0.6% larger.
// (Forcing the sequential elision's parallel_for the same way cost PBBS
// invertedIndex 7.5% there, so that one is not forced.)
#if defined(__clang__)
#define SPORK_LOOP_BODY_CALL [[clang::always_inline]]
#else
#define SPORK_LOOP_BODY_CALL
#endif

namespace parlay {
namespace internal {

// A lambda capture that copies a value when copying is free (trivially
// copyable and at most copy_cap bytes) and otherwise refers to it, like [&].
// Use it as an init-capture and read it by calling it:
//
//   parallel_for(0, n, [buffer, v = cap(v)](size_t i) { buffer[i] = v(); });
//
// A copy lives in the closure rather than behind a reference, so a store the
// loop body makes (a byte store may alias anything) cannot clobber it: it can
// stay in a register, and the loop can vectorize.  Either form compiles to the
// same code as writing [v] or [&v] by hand.  A copy is a snapshot taken when
// the lambda is made, so only cap values the loop does not change.  A copy is
// read through a const reference, so a capped callable is called as const.
template <typename T, bool by_value>
struct capped_copy;

template <typename T>
struct capped_copy<T, true> {
  std::remove_cv_t<T> value;
  const std::remove_cv_t<T>& operator()() const noexcept { return value; }
};

template <typename T>
struct capped_copy<T, false> {
  T* ref;
  T& operator()() const noexcept { return *ref; }
};

template <size_t copy_cap = 16, typename T>
constexpr auto cap(T& x) noexcept {
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_trivially_copyable_v<U> && sizeof(U) <= copy_cap)
    return capped_copy<T, true>{x};
  else
    return capped_copy<T, false>{std::addressof(x)};
}

// The cap for loop bodies passed down through parallel_for's wrappers.  Loop
// bodies that capture by reference are a few pointers; each wrapper level
// adds its own few words around the body it holds.
inline constexpr size_t functor_copy_cap = 128;

// Caps a loop body f that is called with Args.  A body that cannot be called
// as const (a mutable lambda) is referenced, since a copy would be read
// through a const reference.
template <typename... Args, typename F>
constexpr auto cap_body(F& f) noexcept {
  if constexpr (std::is_invocable_v<const std::remove_cv_t<F>&, Args...>)
    return cap<functor_copy_cap>(f);
  else
    return capped_copy<F, false>{std::addressof(f)};
}

}  // namespace internal
}  // namespace parlay

#endif  // PARLAY_INTERNAL_CAPPED_COPY_H_
