#ifndef PARLAY_INTERNAL_ASYNC_SIGNAL_SAFE_POINTER_H_
#define PARLAY_INTERNAL_ASYNC_SIGNAL_SAFE_POINTER__H
#include <atomic>

template <typename T>
class async_signal_safe_pointer {
  static_assert(std::atomic<T*>::is_always_lock_free,
                "async_signal_safe_pointer relies on being always lock free!");
  private:
  std::atomic<T*> ptr;
  public:

  inline void store(T* p) noexcept {
    std::atomic_signal_fence(std::memory_order_release);
    ptr.store(p, std::memory_order_relaxed);
  }

  inline T* load() const noexcept {
    T* p = ptr.load(std::memory_order_relaxed);
    std::atomic_signal_fence(std::memory_order_acquire);
    return p;
  }

  inline constexpr async_signal_safe_pointer() noexcept : ptr(nullptr) {}

  inline async_signal_safe_pointer(T* p) noexcept {
    store(p);
  }

  inline T& operator*() {
    return *load();
  }

  inline const T& operator*() const {
    return *load();
  }

  inline T* operator->() noexcept {
    return load();
  }

  inline const T* operator->() const noexcept {
    return load();
  }

  inline async_signal_safe_pointer<T>& operator=(async_signal_safe_pointer<T>&& other) noexcept {
    store(other.load());
    return *this;
  }

  inline bool operator==(async_signal_safe_pointer<T>&& other) const noexcept {
    return load() == other.load();
  }
};

#endif
