#pragma once

#include <chrono>
#include <string>
#include<functional>

class ScopedTimer {
    public:
                using Clock = std::chrono::steady_clock;
                using Callback = std::function<void(const std::string&, double)>;
                /**
                 * @brief Starts a timer for the specified stage.
                 *
                 * The elapsed time is reported through the supplied callback when
                 * the ScopedTimer goes out of scope.
                 *
                 * @param stage_name Name of the stage being measured.
                 * @param callback Callback invoked with the stage name and elapsed
                 *                 time in milliseconds.
                 */
                explicit ScopedTimer(
                        std::string stage_name,
                        Callback callback
                );
                /**
                 * @brief Stops the timer and reports the elapsed duration.
                 *
                 * The destructor is noexcept because timing/reporting must never
                 * allow an exception to escape an RAII destructor.
                 */

                 ~ScopedTimer() noexcept;

                 ScopedTimer(const ScopedTimer&) = delete;
                 ScopedTimer& operator=(const ScopedTimer&) = delete;
                 
                 ScopedTimer(ScopedTimer&&) = delete;
                 ScopedTimer& operator=(const ScopedTimer&&) = delete;
private:
                std::string name_;
                Clock::time_point start_;
                Callback callback_;                

};