#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NUMBER="${1:-3}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/runs/${RUN_NUMBER}/${RUN_ID}"
CSV_PATH="${RUN_DIR}/benchmark.csv"

mkdir -p "${RUN_DIR}"

echo "Saving benchmark to ${RUN_DIR}"

cmake --build build/release --target timing_benchmark -j"$(nproc)"

# Save the exact source revision and working-tree state.
git rev-parse HEAD > "${RUN_DIR}/git-commit.txt"

# Record system and build information.
{
    echo "Timestamp UTC: ${RUN_ID}"
    echo
    echo "=== Operating system ==="
    uname -v -m -o
    echo
    echo "=== CMake ==="
    cmake --version
    echo
    echo "=== C++ compiler ==="
    c++ --version
    echo
    echo "=== Build configuration ==="
    grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):' \
        build-release/CMakeCache.txt || true
} > "${RUN_DIR}/metadata.txt" 2>&1

# Save output while displaying it in the terminal.
set +e
taskset -c 0 ./build/release/timing_benchmark --warmup 10 --iterations 1000 \
    --output "${CSV_PATH}" \
    2>&1 | tee "${RUN_DIR}/output.txt"
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e

echo "${BENCHMARK_STATUS}" > "${RUN_DIR}/exit-code.txt"

if [[ "${BENCHMARK_STATUS}" -ne 0 ]]; then
    echo "Benchmark failed with exit code ${BENCHMARK_STATUS}" >&2
    exit "${BENCHMARK_STATUS}"
fi

echo "Benchmark completed. Results: ${RUN_DIR}"
