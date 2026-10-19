# Average Benchmark Results: Run 3

Averages across 3 CSV runs: benchmark, benchmark, benchmark.

All latency values are in milliseconds. Stages are averaged only across runs that contain the stage.

| Stage | Runs | Samples | Total (ms) | Average (ms) | Min (ms) | Max (ms) | Stddev (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capture_and_preprocess | 3 | 1000 | 28.150 | 0.028 | 0.017 | 0.299 | 0.025 | 0.023 | 0.033 | 0.049 | 0.147 |
| inference_pipeline | 3 | 1000 | 16949.377 | 16.949 | 13.287 | 40.792 | 1.330 | 16.780 | 17.539 | 18.122 | 21.216 |
| postprocess | 3 | 1000 | 0.899 | 0.001 | 0.000 | 0.111 | 0.005 | 0.001 | 0.001 | 0.001 | 0.001 |
| total_ms | 3 | 1000 | 16978.499 | 16.978 | 13.309 | 40.821 | 1.332 | 16.807 | 17.569 | 18.150 | 21.247 |
| triton_inference | 3 | 1000 | 16571.877 | 16.572 | 12.953 | 40.225 | 1.311 | 16.405 | 17.153 | 17.721 | 20.716 |
