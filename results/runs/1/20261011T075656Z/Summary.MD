## Preliminary comparison: Run 1 vs. Run 2

Run 2 recorded a pipeline mean latency of 21.958 ms, compared with 24.945 ms in Run 1, a reduction of approximately 12.0%. Median latency decreased by 16.0%, and P95 latency decreased by 8.6%.

Mean latency also decreased in Triton inference (−10.6%), preprocessing (−17.0%), and image loading (−5.4%). However, pipeline standard deviation increased from 3.584 ms to 3.872 ms, and the maximum observed latency increased from 39.086 ms to 41.476 ms.

These observations suggest lower typical latency in Run 2 but do not establish a persistent performance improvement. Additional controlled runs are required to quantify run-to-run variation and determine whether the observed differences are reproducible.

No implementation change or causal explanation is attributed to the difference between runs.
