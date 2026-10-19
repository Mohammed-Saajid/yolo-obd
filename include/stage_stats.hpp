#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <string>
#include <unordered_map>


class StageStats final {
    public:
    StageStats() = default;
    
    void record(double elapsed_ms);
    
    [[nodiscard]] std::size_t samples() const noexcept;
    
    [[nodiscard]] double total_ms() const noexcept;
    
    [[nodiscard]] std::optional<double> min_ms() const noexcept;
    [[nodiscard]] std::optional<double> max_ms() const noexcept;
    [[nodiscard]] std::optional<double> average_ms() const noexcept;
    
    [[nodiscard]] std::optional<double> variance_ms2() const noexcept;
    [[nodiscard]] std::optional<double> stddev_ms() const noexcept;
    
    [[nodiscard]] std::optional<double> percentile_ms(
        double percentile
    ) const ;
    
    private:
    std::size_t samples_ = 0;
    double total_ms_ = 0.0;
    double mean_ms_ = 0.0;
    double m2_ms2_ = 0.0;
    double min_ms_ = 0.0;
    double max_ms_ = 0.0;
    
    std::vector<double> observations_;
    
};

using StageStatsMap  = std::unordered_map<std::string, StageStats>;
void print_summary(const StageStatsMap &stats_map)  noexcept;