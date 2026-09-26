#!/bin/bash
# Evaluate OpenUnlearning's released TOFU checkpoint on the forget10 split. Nothing is trained.
# Model: open-unlearning/tofu_Llama-3.2-1B-Instruct_full (ungated; Meta's base model is gated).
# About 7 minutes on a DGX Spark.
#
# Differences from the README's evaluation command:
#   setup_data.py --eval_logs     the README says --eval, which is not a real flag. This
#                                 downloads the published eval logs, including the retain90
#                                 logs that forget_quality and privleak are computed against.
#   attn_implementation=sdpa      every configs/model/*.yaml hardcodes flash_attention_2,
#                                 which isn't installed (no Blackwell aarch64 build).
#   torch_dtype=float32           the configs ask for bfloat16, but evals/metrics/utils.py calls
#                                 .cpu().numpy() on the loss tensor and numpy has no bfloat16.
#                                 fp32 avoids patching their code; a 1B model is ~5 GB.
#   tokenizer path override       the README overrides only the model path, which leaves the
#                                 tokenizer pointing at gated meta-llama/* and the run fails
#                                 with a 401. The released checkpoint ships its own tokenizer.
set -u

OU_DIR=${OU_DIR:-$HOME/open-unlearning}
VENV=${VENV:-$HOME/envs/openunlearning}
model=Llama-3.2-1B-Instruct

source "$VENV/bin/activate"
cd "$OU_DIR"
echo "=== start $(date -Is) ==="
python setup_data.py --eval_logs 2>&1 | tail -5

python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${model} \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
  model.model_args.attn_implementation=sdpa \
  model.model_args.torch_dtype=float32 \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
  retain_logs_path=saves/eval/tofu_${model}_retain90/TOFU_EVAL.json \
  task_name=tofu_full_eval
echo "=== exit $? at $(date -Is) ==="
