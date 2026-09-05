# AAG scale-up image for the EKS B200 training clusters (sm_100 -> CUDA 12.8 wheels).
#
# odytrain's torchrun mode runs, inside the image's WORKDIR:
#     uv run --no-sync torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS ... <launcher_script>
# so WORKDIR must be a uv project with a populated .venv. /app/aag is that project;
# the launcher is cluster/entry.py, which reads the stage list from the YAML odytrain
# mounts at /etc/config/config.yaml (--config_path).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    HF_HOME=/mnt/shared/aag/hf_home TORCH_HOME=/mnt/shared/aag/torch_home
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv git curl ca-certificates libgl1 libglib2.0-0 iproute2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/
WORKDIR /app/aag
COPY pyproject.toml ./
RUN uv venv --python 3.12 .venv \
 && uv pip install --python .venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
 && uv pip install --python .venv/bin/python \
      lpips diffusers accelerate pyarrow pillow numpy scipy pytorch-fid \
      "huggingface_hub[hf_transfer]" pyyaml nltk
COPY aag ./aag
COPY scripts ./scripts
COPY cluster ./cluster
RUN uv pip install --python .venv/bin/python --no-deps -e . \
 && .venv/bin/python -c "import torch, lpips, diffusers, pyarrow; print(torch.__version__)"
ENV PATH="/app/aag/.venv/bin:$PATH" PYTHONPATH=/app/aag

# gcc for torch.compile (Triton/Inductor needs a C compiler); appended last so the heavy layers stay cached
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev python3-dev && rm -rf /var/lib/apt/lists/*
