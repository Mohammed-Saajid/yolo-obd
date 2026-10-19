sudo docker run --gpus all --ipc=host --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/model_repository/triton_serve:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models --log-verbose=1 \
  --allow-metrics=true --allow-gpu-metrics=true \
  --metrics-interval-ms=1000
