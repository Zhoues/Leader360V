# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 vllm serve /share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct \
#   --tensor-parallel-size 1 \
#   --limit-mm-per-prompt.video 0 \
#   --port 25547 \
  # --gpu-memory-utilization 0.5 \

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 python -m vllm.entrypoints.openai.api_server \
  --model /share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --port 25547