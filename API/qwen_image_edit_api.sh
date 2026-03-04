CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 vllm serve /share/project/zhouenshen/hpfs/ckpt/diffusion/Qwen-Image-Edit-2511 \
  --omni \
  --port 25548 \
  --tensor-parallel-size 1


# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python -m vllm.entrypoints.openai.api_server \
#   --model /share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct \
#   --tensor-parallel-size 1 \
#   --limit-mm-per-prompt.video 0 \
#   --port 25547