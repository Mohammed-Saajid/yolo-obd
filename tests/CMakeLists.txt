# Fetch or find GoogleTest
find_package(GTest REQUIRED)

add_executable(stage_stats_tests
    stage_stats_test.cpp
)

target_link_libraries(stage_stats_tests
    PRIVATE
        stage_stats
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(stage_stats_tests)