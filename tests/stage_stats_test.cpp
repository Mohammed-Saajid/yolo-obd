
#include "stage_stats.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

TEST(StageStatsTest, CalculatesBasicStatistics) {
    StageStats stats;

    stats.record(10.0);
    stats.record(20.0);
    stats.record(30.0);

    EXPECT_EQ(stats.samples(), 3U);
    EXPECT_DOUBLE_EQ(stats.total_ms(), 60.0);
    ASSERT_TRUE(stats.average_ms().has_value());
    EXPECT_DOUBLE_EQ(*stats.average_ms(), 20.0);

    ASSERT_TRUE(stats.min_ms().has_value());
    EXPECT_DOUBLE_EQ(*stats.min_ms(), 10.0);

    ASSERT_TRUE(stats.max_ms().has_value());
    EXPECT_DOUBLE_EQ(*stats.max_ms(), 30.0);
}

TEST(StageStatsTest, CalculatesSampleVarianceAndStddev) {
    StageStats stats;

    stats.record(10.0);
    stats.record(20.0);
    stats.record(30.0);

    // Sample variance = 100.0 ms^2.
    ASSERT_TRUE(stats.variance_ms2().has_value());
    EXPECT_DOUBLE_EQ(*stats.variance_ms2(), 100.0);

    ASSERT_TRUE(stats.stddev_ms().has_value());
    EXPECT_NEAR(*stats.stddev_ms(), 10.0, 1e-12);
}

TEST(StageStatsTest, CalculatesInterpolatedPercentiles) {
    StageStats stats;

    stats.record(10.0);
    stats.record(20.0);
    stats.record(30.0);

    ASSERT_TRUE(stats.percentile_ms(50.0).has_value());
    EXPECT_DOUBLE_EQ(*stats.percentile_ms(50.0), 20.0);

    ASSERT_TRUE(stats.percentile_ms(25.0).has_value());
    EXPECT_DOUBLE_EQ(*stats.percentile_ms(25.0), 15.0);

    ASSERT_TRUE(stats.percentile_ms(100.0).has_value());
    EXPECT_DOUBLE_EQ(*stats.percentile_ms(100.0), 30.0);
}

TEST(StageStatsTest, HandlesEmptyAndSingleSampleCases) {
    StageStats empty;

    EXPECT_EQ(empty.samples(), 0U);
    EXPECT_FALSE(empty.average_ms().has_value());
    EXPECT_FALSE(empty.min_ms().has_value());
    EXPECT_FALSE(empty.variance_ms2().has_value());
    EXPECT_FALSE(empty.percentile_ms(50.0).has_value());

    StageStats single;
    single.record(12.0);

    ASSERT_TRUE(single.average_ms().has_value());
    EXPECT_DOUBLE_EQ(*single.average_ms(), 12.0);
    EXPECT_FALSE(single.variance_ms2().has_value());
    EXPECT_FALSE(single.stddev_ms().has_value());
}

TEST(StageStatsTest, RejectsInvalidMeasurements) {
    StageStats stats;

    EXPECT_THROW(
        stats.record(-1.0),
        std::invalid_argument
    );

    EXPECT_THROW(
        stats.record(std::numeric_limits<double>::infinity()),
        std::invalid_argument
    );

    EXPECT_THROW(
        stats.record(std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument
    );

    EXPECT_EQ(stats.samples(), 0U);
}

TEST(StageStatsTest, RejectsInvalidPercentiles) {
    StageStats stats;
    stats.record(10.0);

    EXPECT_THROW(
        stats.percentile_ms(-1.0),
        std::invalid_argument
    );

    EXPECT_THROW(
        stats.percentile_ms(101.0),
        std::invalid_argument
    );
}
