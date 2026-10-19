# Average Benchmark Results: File input (without camera)

Averages across 3 CSV runs: 20261018T061054Z, 20261018T061358Z, 20261018T061614Z.

All latency values are in milliseconds. Stages are averaged only across runs that contain the stage.

| Stage | Runs | Samples | Total (ms) | Average (ms) | Min (ms) | Max (ms) | Stddev (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Total time | 3 | 1000 | 29830.791 | 29.831 | 22.754 | 142.859 | 7.433 | 27.635 | 37.089 | 43.421 | 56.587 |
| image_load | 3 | 1000 | 1966.252 | 1.966 | 0.565 | 13.433 | 0.868 | 1.840 | 2.615 | 2.928 | 4.755 |
| inference_pipeline | 3 | 1000 | 27857.275 | 27.857 | 21.195 | 137.127 | 7.164 | 25.759 | 34.709 | 40.126 | 54.162 |
| postprocess | 3 | 1000 | 1.289 | 0.001 | 0.001 | 0.145 | 0.005 | 0.001 | 0.001 | 0.002 | 0.004 |
| preprocess | 3 | 1000 | 5225.343 | 5.225 | 2.271 | 37.616 | 1.915 | 5.020 | 6.192 | 6.922 | 12.286 |
| triton_inference | 3 | 1000 | 22219.830 | 22.220 | 18.097 | 126.526 | 6.368 | 20.433 | 27.876 | 32.466 | 45.187 |
