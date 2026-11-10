#ifndef PARLAY_INTERNAL_SPORK_DEQUE_H_
#define PARLAY_INTERNAL_SPORK_DEQUE_H_

// Promotable forks: each stack's list of open forks (its spork deque), the
// frontier up to which they are promoted, and promoting them, in a beat or
// eagerly.

#include "spork_timing.h"
#include "spork_tokens.h"
#include "../portability.h"

#include <atomic>

namespace spork {

template <typename T>
class sigsafe_ptr {
  static_assert(__atomic_always_lock_free(sizeof(T*), 0),
                "sigsafe_ptr relies on being always lock free!");
 private:
  // Accessed only atomically: relaxed, behind signal fences.
  T* ptr;
 public:
  inline void store(T* p) noexcept {
    std::atomic_signal_fence(std::memory_order_release);
    __atomic_store_n(&ptr, p, __ATOMIC_RELAXED);
  }

  inline T* load() const noexcept {
    T* p = __atomic_load_n(&ptr, __ATOMIC_RELAXED);
    std::atomic_signal_fence(std::memory_order_acquire);
    return p;
  }

  inline constexpr sigsafe_ptr() noexcept : ptr(nullptr) {}

  // Not copyable: a copy would bypass load and store.
  sigsafe_ptr(const sigsafe_ptr&) = delete;
  sigsafe_ptr& operator=(const sigsafe_ptr&) = delete;

  // x = p is x.store(p).
  inline sigsafe_ptr& operator=(T* p) noexcept {
    store(p);
    return *this;
  }
};

// A promotable fork: pushed onto its stack's list for the duration of the
// fork's left side (spork_open, then close), oldest first.  The fork that
// owns it derives from it, with whatever its promotion needs (see par and
// parfor_).
// A slot has no promoted flag: it is promoted exactly when it is the
// frontier or older (see spork_deque_frontier).
struct SporkSlot {
  // The promotion, called with this slot.  A plain function, which knows
  // what the slot belongs to, rather than a closure object: pushing a slot
  // writes one pointer for it.
  void (*promfn)(SporkSlot*);
  SporkSlot* prev;   // the next older slot, or the list's sentinel
  // The next newer slot, if any.  Left unwritten until that slot links in:
  // promote_front reads it only for slots older than spork_deque_back,
  // which is this slot until then.  So, like prev, it needs no atomic
  // access: link writes it while promote_front won't read it, and link's
  // store to spork_deque_back (a release) publishes it.
  SporkSlot* next;

  // A list's sentinel (spork_deque_front, Fiber::deque_front).
  constexpr explicit SporkSlot() :
    promfn(nullptr), prev(nullptr), next(nullptr) {}

  // Not yet on the list: its owner finishes building it, and spork_open
  // links it.
  explicit SporkSlot(void (*fn)(SporkSlot*)) noexcept : promfn(fn) {}

  void link() noexcept;
  bool close();
  void promote();
  static void promote_front();
  void eager_promote();
};

// Slots live in the frames of the stack that created them, so the list
// head belongs to that stack too: a fiber owns its own sentinel, and
// spork_deque_front is the sentinel for a thread's own (pinned) stack.
// spork_deque_back is the newest slot on the list in use on this thread
// (its sentinel if the list is empty, null until spork_thread_init); it and
// the frontier are saved and restored across a handoff.
inline constinit thread_local SporkSlot spork_deque_front{};
inline constinit thread_local sigsafe_ptr<SporkSlot> spork_deque_back;
// The newest promoted slot on the list in use, or its sentinel if none is.
// Promotions go oldest first and are never undone, so every slot up to it
// is promoted and every slot after it is not: a beat starts promoting just
// after it.  Promotions move it forward and a promoted slot's close moves
// it back; linking a slot and closing an unpromoted one never touch it.  A
// closing slot is the newest, so it is promoted exactly when it is the
// frontier.
// Saved and restored with the back pointer.
inline constinit thread_local sigsafe_ptr<SporkSlot> spork_deque_frontier;

// Make this the newest slot.  Everything its promotion reads must be set
// first: from here on the heartbeat handler may promote it.
inline void SporkSlot::link() noexcept {
  prev = spork_deque_back.load();
  prev->next = this;
  spork_deque_back = this;
}

inline bool SporkSlot::close() {
  // This slot, the newest, read back from the list rather than kept from
  // before the fork's left side ran: comparing against `this` keeps the
  // slot's address in a callee-saved register across the left side.
  SporkSlot* const self = spork_deque_back.load();
  spork_deque_back = prev;
  // Read the frontier only once unlinked: read before, it could miss a beat
  // that promotes this slot in between, whose job would never be synced.
  std::atomic_signal_fence(std::memory_order_seq_cst);
  if (spork_deque_frontier.load() != self) [[likely]] return false;
  // Promoted: the frontier moves back to the slot before it.  A beat that
  // comes in between finds the frontier already unlinked (promote_front).
  spork_deque_frontier = prev;
  return true;
}

inline void SporkSlot::promote() {
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->promotions, 1);
#endif
  heartbeat_tokens = heartbeat_tokens - 1;
  promfn(this);
}

inline void SporkSlot::promote_front() {
  SporkSlot* slot = spork_deque_frontier.load();
  SporkSlot* const back = spork_deque_back.load();
  // The frontier's close has unlinked it but not yet moved the frontier
  // back: it was the newest slot, so every slot left is promoted.
  if (slot->prev == back) return;
  while (heartbeat_tokens && slot != back) {
    slot = slot->next;
    slot->promote();
    spork_deque_frontier = slot;
  }
}

// A thread's own stack gets an empty spork deque of its own.
inline void spork_thread_init() noexcept {
  if (spork_deque_back.load() == nullptr) {
    spork_deque_back = &spork_deque_front;
    spork_deque_frontier = &spork_deque_front;
  }
}

PARLAY_NOINLINE inline void SporkSlot::eager_promote() {
#ifdef SPORK_HEARTBEAT_TIMING
  const std::uint64_t timing_start = __rdtsc();
#endif
  const HeartbeatsDisabled beats_off;
  // A beat since link may have promoted it already, making it the frontier.
  if (heartbeat_tokens && spork_deque_frontier.load() != this) {
    promote();
    spork_deque_frontier = this;   // the newest slot: with tokens to spend, every older one is promoted
  }
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->eager_ticks, __rdtsc() - timing_start);
#endif
}

// Open a promotable fork whose slot is `slot` (built, not yet linked): link
// it, and promote it at once if this computation holds tokens.  The fork
// then runs its left side and closes the slot, which says whether it was
// promoted (see par and parfor_, which also handle a left side that throws).
PARLAY_INLINE void spork_open(SporkSlot& slot) {
  slot.link();
  if (heartbeat_tokens) [[unlikely]]
    slot.eager_promote();
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_DEQUE_H_
