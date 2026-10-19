#include "stage_stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include  <iostream>
#include <iomanip>


void StageStats::record(double elapsed_ms) {
    if (!std::isfinite(elapsed_ms) || elapsed_ms < 0.0) {
        throw std::invalid_argument(
            "Elapsed time must be finite and non-negative."
        );
    }

    if (samples_ == std::numeric_limits<std::size_t>::max() ) {
        throw std::overflow_error("StageStats sample count overflow");
    }

    observations_.push_back(elapsed_ms);

    ++samples_;

    total_ms_+=elapsed_ms;
    
    if (samples_==1) {
        min_ms_ = elapsed_ms;
        max_ms_ = elapsed_ms;
        mean_ms_ = elapsed_ms;
        m2_ms2_ = 0.0;
        return;
    }

    min_ms_ = std::min(min_ms_, elapsed_ms);
    max_ms_ = std::max(max_ms_, elapsed_ms);


    // Welford's  Online Mean and  Sum of Squared Deviations
    const double delta = elapsed_ms - mean_ms_;
    mean_ms_ += delta / static_cast<double>(samples_);
    const double delta_after_update = elapsed_ms - mean_ms_;
    m2_ms2_ += delta * delta_after_update;
}

std::size_t StageStats::samples() const noexcept{
    return samples_;
}

double StageStats::total_ms() const noexcept {
    return total_ms_;
}

std::optional<double> StageStats::min_ms() const noexcept {
    if (samples_ == 0) {
        return std::nullopt;
    }
    return min_ms_;
}

std::optional<double> StageStats::max_ms() const noexcept {
    if (samples_ == 0) {
        return std::nullopt;
    }
    return max_ms_;
}

std::optional<double> StageStats::average_ms() const noexcept {
    if (samples_ == 0) {
        return std::nullopt;
    }
    return mean_ms_;
}

std::optional<double> StageStats::variance_ms2() const noexcept {
    if (samples_ < 2) {
        return std::nullopt;
    }
    return m2_ms2_ / static_cast<double>(samples_-1);
}

std::optional<double> StageStats::stddev_ms() const noexcept {
    const auto variance = variance_ms2();

    if (!variance.has_value()) {
        return std::nullopt;
    }

    return std::sqrt(*variance);
}

std::optional<double> StageStats::percentile_ms(
    double percentile
) const {
    if (!std::isfinite(percentile) ||
        percentile < 0.0 ||
        percentile > 100.0) {
        throw std::invalid_argument(
            "Percentile must be between 0 and 100."
        );
    }

    if (observations_.empty()) {
        return std::nullopt;
    }

    std::vector<double> sorted = observations_;
    std::sort(sorted.begin(), sorted.end());

    if (sorted.size() == 1) {
        return sorted.front();
    }

    // Linear interpolation over the [0, N - 1] index range.
    const double position =
        (percentile / 100.0) *
        static_cast<double>(sorted.size() - 1);

    const auto lower_index =
        static_cast<std::size_t>(position);

    const auto upper_index =
        std::min(lower_index + 1, sorted.size() - 1);

    const double fraction =
        position - static_cast<double>(lower_index);

    return sorted[lower_index] +
        fraction * (sorted[upper_index] - sorted[lower_index]);
}


void print_stat(
    const char* label,
    const std::optional<double>& value,
    const char* unit
) {
    std::cout << "  " << std::left << std::setw(16) << label;

    if (value.has_value()) {
        std::cout << std::right << std::fixed
                  << std::setprecision(3) << *value
                  << ' ' << unit;
    } else {
        std::cout << "N/A";
    }

    std::cout << '\n';
}

void print_summary(const StageStatsMap& stats_map)  noexcept {
    for (const auto& [stage, stats] : stats_map) {
        std::cout << "\nStage: " << stage << '\n'
                  << "  Samples:         " << stats.samples() << '\n';

        print_stat("Total", stats.total_ms(), "ms");
        print_stat("Average", stats.average_ms(), "ms");
        print_stat("Minimum", stats.min_ms(), "ms");
        print_stat("Maximum", stats.max_ms(), "ms");
        print_stat("Variance", stats.variance_ms2(), "ms^2");
        print_stat("Std deviation", stats.stddev_ms(), "ms");
        print_stat("P50", stats.percentile_ms(50.0), "ms");
        print_stat("P90", stats.percentile_ms(90.0), "ms");
        print_stat("P95", stats.percentile_ms(95.0), "ms");
        print_stat("P99", stats.percentile_ms(99.0), "ms");
    }
}
