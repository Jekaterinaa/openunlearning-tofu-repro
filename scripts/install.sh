#!/bin/bash
# Install OpenUnlearning on aarch64 + Blackwell (DGX Spark) in its own venv.
#
# OpenUnlearning's requirements.txt hard-pins torch==2.4.1, numpy, transformers and more, so it
# gets a venv of its own rather than sharing one with other projects.
#
# Deviations from the README, all forced by aarch64 + Blackwell:
#   torch==2.4.1  -> PyTorch from the cu130 index instead (the pin has no working wheel here)
#   bitsandbytes  -> skipped: no aarch64 wheel, and it is only needed for quantisation
#   flash-attn    -> skipped: no Blackwell aarch64 build; runs use sdpa attention instead
#   deepspeed     -> installed with DS_BUILD_OPS=0. It looks training-only, but
#                    src/trainer/__init__.py imports RMU unconditionally and rmu.py imports
#                    deepspeed at module level, so every entry point (eval included) fails
#                    without it. With ops disabled it builds cleanly on ARM.
#
# On x86 + CUDA 12 set TORCH_INDEX=https://download.pytorch.org/whl/cu121 (or whatever
# matches your driver).
set -euo pipefail

OU_DIR=${OU_DIR:-$HOME/open-unlearning}
VENV=${VENV:-$HOME/envs/openunlearning}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}

cd "$OU_DIR"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

echo "=== 1. torch ==="
pip install torch --index-url "$TORCH_INDEX" 2>&1 | tail -3
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== 2. everything else, minus the pins that break on ARM ==="
grep -vE '^(torch|deepspeed|bitsandbytes)==' requirements.txt > /tmp/ou_req.txt
cat /tmp/ou_req.txt
pip install -r /tmp/ou_req.txt 2>&1 | tail -5

echo "=== 3. deepspeed, without compiled ops ==="
DS_BUILD_OPS=0 pip install deepspeed==0.15.4 2>&1 | tail -3

echo "=== 4. the package itself, no deps (already resolved above) ==="
pip install --no-deps -e . 2>&1 | tail -3
pip install -q "lm-eval==0.4.11" 2>&1 | tail -3

echo "=== 5. verify ==="
python -c "import torch, transformers, hydra, deepspeed; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo "=== done at $(date -Is) ==="
