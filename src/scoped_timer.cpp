#include "scoped_timer.hpp"

#include <utility>

ScopedTimer::ScopedTimer(
    std::string stage_name,
    Callback callback
) 
        : name_(std::move(stage_name)),
            start_(Clock::now()),
            callback_(std::move(callback)) {}

ScopedTimer::~ScopedTimer() noexcept {
    const auto end = Clock::now();

    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        end - start_
    ).count();
    if (!callback_) {
        return;
    }

    try {
        callback_(name_, elapsed_ms);
    } catch (...) {}
}