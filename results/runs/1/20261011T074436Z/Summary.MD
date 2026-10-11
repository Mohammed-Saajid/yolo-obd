# YOLO26n Inference Pipeline — Baseline Run 001

## Summary

The inference pipeline recorded a mean latency of 24.945 ms across 100 samples, with a median of 24.122 ms and a P95 of 33.212 ms.

Triton inference had the largest measured mean stage duration at 20.700 ms, followed by preprocessing at 3.863 ms and image loading at 1.132 ms. Postprocessing contributed a negligible measured duration in this run.

## Initial observations

- Triton inference is the first stage to investigate for latency optimization.
- Pipeline tail latency is meaningfully higher than median latency.
- Component-stage means exceed the pipeline mean, so timing boundaries and possible overlap require verification.
- Results are preliminary and come from one run.

## Conclusion

This run establishes an initial measurement baseline. No performance improvement or regression can be claimed until comparable repeated runs have been collected and analyzed.
