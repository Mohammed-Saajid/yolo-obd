sudo docker run -it --rm --net=host --gpus all \
  nvcr.io/nvidia/tritonserver:24.05-py3-sdk \
  perf_analyzer -m yolo26n -u localhost:8001 -i grpc