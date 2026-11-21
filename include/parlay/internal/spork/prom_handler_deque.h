#ifndef PARLAY_INTERNAL_SPORK_PROM_HANDLER_DEQUE_H_
#define PARLAY_INTERNAL_SPORK_PROM_HANDLER_DEQUE_H_

// Promotable forks: each stack's list of open forks (its prom handler
// deque), the last of them promoted, and promoting them, in a beat or
// eagerly.

#include "timing.h"
#include "tokens.h"
#include "../../portability.h"

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
// fork's left side (open, then close), oldest first.  The fork that owns it
// derives from it, with whatever its promotion needs (see par and parfor_).
// A slot has no promoted flag: it is promoted exactly when it is the last
// promoted slot or older (see prom_handler_last).
struct PromHandler {
  // The promotion, called with this slot.  A plain function, which knows
  // what the slot belongs to, rather than a closure object: pushing a slot
  // writes one pointer for it.
  void (*promfn)(PromHandler*);
  PromHandler* prev;   // the next older slot, or the list's sentinel
  // The next newer slot, if any.  Left unwritten until that slot links in:
  // promote_after_last reads it only for slots older than
  // prom_handler_newest, which is this slot until then.  So, like prev, it
  // needs no atomic access: link writes it while promote_after_last won't
  // read it, and link's store to prom_handler_newest (a release) publishes
  // it.
  PromHandler* next;

  // A list's sentinel (prom_handler_sentinel, Fiber::sentinel).
  constexpr explicit PromHandler() :
    promfn(nullptr), prev(nullptr), next(nullptr) {}

  // Not yet on the list: its owner finishes building it, and open links it.
  explicit PromHandler(void (*fn)(PromHandler*)) noexcept : promfn(fn) {}

  void open();
  bool close();
  static void promote_after_last();

 private:
  void link() noexcept;
  void promote();
  void eager_promote();
};

// Slots live in the frames of the stack that created them, so the list
// head belongs to that stack too: a fiber owns its own sentinel, and
// prom_handler_sentinel is the sentinel for a thread's own (pinned) stack.
// prom_handler_newest is the newest slot on the list in use on this thread
// (its sentinel if the list is empty, null until spork_thread_init); it and
// prom_handler_last are saved and restored across a handoff.
inline constinit thread_local PromHandler prom_handler_sentinel{};
inline constinit thread_local sigsafe_ptr<PromHandler> prom_handler_newest;
// The newest promoted slot on the list in use, or its sentinel if none is.
// Promotions go oldest first and are never undone, so every slot up to it
// is promoted and every slot after it is not: a beat starts promoting just
// after it.  Promotions move it forward and a promoted slot's close moves
// it back; linking a slot and closing an unpromoted one never touch it.  A
// closing slot is the newest, so it is promoted exactly when it is the
// last promoted slot.
// Saved and restored with prom_handler_newest.
inline constinit thread_local sigsafe_ptr<PromHandler> prom_handler_last;

// Make this the newest slot.  Everything its promotion reads must be set
// first: from here on the heartbeat handler may promote it.
inline void PromHandler::link() noexcept {
  prev = prom_handler_newest.load();
  prev->next = this;
  prom_handler_newest = this;
}

inline bool PromHandler::close() {
  // This slot, the newest, read back from the list rather than kept from
  // before the fork's left side ran: comparing against `this` keeps the
  // slot's address in a callee-saved register across the left side.
  PromHandler* const self = prom_handler_newest.load();
  prom_handler_newest = prev;
  // Read the last promoted slot only once unlinked: read before, it could
  // miss a beat that promotes this slot in between, whose job would never be
  // synced.
  std::atomic_signal_fence(std::memory_order_seq_cst);
  if (prom_handler_last.load() != self) [[likely]] return false;
  // Promoted: the last promoted slot is now the one before it.  A beat that
  // comes in between finds the last promoted slot already unlinked
  // (promote_after_last).
  prom_handler_last = prev;
  return true;
}

inline void PromHandler::promote() {
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->promotions, 1);
#endif
  heartbeat_tokens = heartbeat_tokens - 1;
  promfn(this);
}

inline void PromHandler::promote_after_last() {
  PromHandler* slot = prom_handler_last.load();
  PromHandler* const newest = prom_handler_newest.load();
  // The last promoted slot's close has unlinked it but not yet moved
  // prom_handler_last back: it was the newest slot, so every slot left is
  // promoted.
  if (slot->prev == newest) return;
  while (heartbeat_tokens && slot != newest) {
    slot = slot->next;
    slot->promote();
    prom_handler_last = slot;
  }
}

// A thread's own stack gets an empty prom handler deque of its own.
inline void spork_thread_init() noexcept {
  if (prom_handler_newest.load() == nullptr) {
    prom_handler_newest = &prom_handler_sentinel;
    prom_handler_last = &prom_handler_sentinel;
  }
}

PARLAY_NOINLINE inline void PromHandler::eager_promote() {
#ifdef SPORK_HEARTBEAT_TIMING
  const std::uint64_t timing_start = __rdtsc();
#endif
  const HeartbeatsDisabled beats_off;
  // A beat since link may have promoted it already, making it the last
  // promoted slot.
  if (heartbeat_tokens && prom_handler_last.load() != this) {
    promote();
    prom_handler_last = this;   // the newest slot: with tokens to spend, every older one is promoted
  }
#ifdef SPORK_HEARTBEAT_TIMING
  if (HeartbeatTiming* t = heartbeat_timing_mine) timing_add(t->eager_ticks, __rdtsc() - timing_start);
#endif
}

// Open a promotable fork with this slot (built, not yet linked): link it,
// and promote it at once if this computation holds tokens.  The fork then
// runs its left side and closes the slot, which says whether it was
// promoted (see par and parfor_, which also handle a left side that throws).
// Under HBC's scheme (SPORK_USE_HBC_HB) the balance is 0 outside a beat, so
// no fork is promoted eagerly, and none checks.
PARLAY_INLINE void PromHandler::open() {
  link();
  if constexpr (!HBC_HB) {
    if (heartbeat_tokens) [[unlikely]]
      eager_promote();
  }
}

}  // namespace spork

#endif  // PARLAY_INTERNAL_SPORK_PROM_HANDLER_DEQUE_H_
