# Reproducing an OpenUnlearning TOFU evaluation on a DGX Spark

This is a small reproduction of [OpenUnlearning](https://github.com/locuslab/open-unlearning), the LLM unlearning framework and benchmark suite by Dorna et al. [1]. I took one of their released TOFU checkpoints, ran their evaluation pipeline on an NVIDIA DGX Spark (GB10 Grace Blackwell GPU, ARM/aarch64 CPU, CUDA 13), and compared the result with the evaluation logs the authors publish for the same model. TOFU itself is the fictitious-authors unlearning benchmark from Maini et al. [2].

Getting there took more than the README's quickstart, mostly because of the ARM + Blackwell combination. The fixes are all in `scripts/`, and they're explained below so you can skip the debugging.

The result: every metric lands within 0.003 of the published value, and `privleak` matches to 14 decimal places.

## What I ran

- OpenUnlearning at commit [`4ad738a`](https://github.com/locuslab/open-unlearning/tree/4ad738aaf60f6a4385f6e2506d01da99e76c31f3)
- Model: [`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`](https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full), Llama-3.2-1B-Instruct fine-tuned on all of TOFU and not unlearned
- Evaluation: the default TOFU suite on the `forget10` / `holdout10` split, with `forget_quality` and `privleak` computed against the published `retain90` logs

I didn't train anything. Evaluating a released checkpoint is the cheapest way to check that the whole pipeline works before trusting it with your own models. The run took about 7 minutes.

## Installing on ARM + Blackwell

OpenUnlearning hard-pins `torch==2.4.1`, `numpy`, `transformers` and a few others, so it gets its own venv. The pins that don't work on this machine:

| Package | Problem | What `scripts/install.sh` does |
|---|---|---|
| `torch==2.4.1` | no working wheel for aarch64 + Blackwell | installs PyTorch from the cu130 index first, then the rest of the requirements without it, then `pip install --no-deps -e .` |
| `flash-attn==2.6.3` | no Blackwell aarch64 build | skips it and uses `sdpa` attention at run time |
| `bitsandbytes==0.44.1` | no aarch64 wheel, only needed for quantisation | skips it |
| `deepspeed` | looks training-only, but isn't optional | installs it with `DS_BUILD_OPS=0 pip install deepspeed==0.15.4` |

The deepspeed one caught me out. `src/trainer/__init__.py` imports RMU unconditionally and `src/trainer/unlearn/rmu.py` does `import deepspeed` at module level, so every entry point fails without it, `src/eval.py` included. With compiled ops turned off it builds on ARM without any trouble. (`logs/install.log` is from my first install, before I'd found this, so deepspeed isn't in it. I installed it afterwards and it's in `env/pip-freeze.txt`.)

pip prints dependency-conflict warnings about `torch==2.4.1` and the missing `bitsandbytes` afterwards. Both are expected, since those pins were overridden on purpose.

## Running the evaluation

The README's evaluation command doesn't run as written on this setup. `scripts/run_eval.sh` has the working version. The differences:

1. `python setup_data.py --eval_logs`. The README says `--eval`, which isn't a real flag.
2. `model.model_args.attn_implementation=sdpa`. Every file in `configs/model/*.yaml` hardcodes `flash_attention_2`. You need this override on every run without flash-attn, for training as well as evaluation.
3. `model.model_args.torch_dtype=float32`. The configs ask for `bfloat16`, but `src/evals/metrics/utils.py` calls `.cpu().numpy()` on the loss tensor, and numpy has no bfloat16, so the run dies with `TypeError: Got unsupported ScalarType BFloat16`. fp32 avoids patching their code, and a 1B model only needs about 5 GB. For a bigger model it would make more sense to patch that line to `.float().cpu().numpy()`.
4. `model.tokenizer_args.pretrained_model_name_or_path=...`. The README overrides only the model path, so the tokenizer still points at gated `meta-llama/Llama-3.2-1B-Instruct` and the run fails with a 401 even though the checkpoint itself is ungated. The released checkpoints ship their own tokenizer, so I point it at the same repo.

```bash
git clone https://github.com/locuslab/open-unlearning.git ~/open-unlearning
git -C ~/open-unlearning checkout 4ad738a

bash scripts/install.sh      # creates ~/envs/openunlearning
bash scripts/run_eval.sh 2>&1 | tee eval.log

cd ~/open-unlearning
python /path/to/this/repo/scripts/compare_to_published.py \
    saves/eval/tofu_Llama-3.2-1B-Instruct_full/TOFU_SUMMARY.json \
    saves/eval/tofu_full_eval/TOFU_SUMMARY.json
```

Paths and the torch index can be overridden with `OU_DIR`, `VENV` and `TORCH_INDEX`. Useful detail: OpenUnlearning's released `open-unlearning/tofu_*` checkpoints aren't gated, so this whole run works without accepting Meta's Llama licence.

## Results

`setup_data.py --eval_logs` downloads the authors' own logs for this exact model and split, so I could compare directly:

| Metric | Published | Mine | Difference |
|---|---|---|---|
| `privleak` | −99.45739100654656 | −99.45739100654656 | exact, to 14 decimal places |
| `forget_Q_A_Prob` | 0.880482 | 0.880805 | +0.0003 |
| `forget_Q_A_ROUGE` | 0.820117 | 0.819417 | −0.0007 |
| `model_utility` | 0.599153 | 0.598106 | −0.0010 |
| `extraction_strength` | 0.706268 | 0.709248 | +0.0030 |
| `forget_quality` | 3.91e−22 | 1.88e−22 | both ≈ 0 |

My run differs from theirs on purpose in two ways, fp32 instead of bf16 and sdpa attention instead of flash-attention-2, and the largest difference is still 0.003.

The numbers also make sense for this model. The `_full` checkpoint was fine-tuned on all of TOFU and never unlearned, so it should behave like a model that remembers the forget set completely. It does: `forget_Q_A_Prob` is 0.88 and `forget_Q_A_ROUGE` is 0.82 (it reproduces the forget answers), `forget_quality` is about 0 (its truth-ratio distribution is as far as it gets from the retain model's), and `privleak` is about −99.5 (maximum membership leakage). A silently broken pipeline would be unlikely to produce this pattern.

The complete per-sample output is in `results/tofu_Llama-3.2-1B-Instruct_full/TOFU_EVAL.json`.

## Why privleak matches exactly

This surprised me at first. Every continuous metric drifted a little, but `privleak` agreed to 14 decimal places. `privleak` isn't hardware-independent by construction: it computes a Min-K% membership score on the forget set *from my model* and compares it with the retain model's logged value. It agrees exactly because it's AUC-based, and AUC depends only on the ordering of the per-sample scores, not on their values. The fp32-vs-bf16 and attention-kernel differences are far too small to reorder any samples.

More generally, AUC-based efficacy measures turn out to be robust to dtype, attention kernel and hardware, while probability- and ROUGE-based measures move slightly. If you're comparing unlearning results across machines, that's a good reason to prefer AUC for the membership inference side.

## What else is in the framework

A few things I found while reading the code that aren't obvious from the README:

- `src/trainer/unlearn/` has RMU, GradDiff, GradAscent (NegGrad), NPO, SimNPO, DPO, CEU, PDU, SatImp, UNDIAL and WGA.
- The default TOFU eval (`configs/eval/tofu.yaml`) computes `forget_truth_ratio`, `forget_quality`, `forget_Q_A_Prob`, `forget_Q_A_ROUGE`, `model_utility`, `privleak` and `extraction_strength`.
- More metrics are available in the same file but commented out: `mia_min_k_plus_plus`, `mia_min_k`, `mia_loss`, `mia_zlib`, `mia_gradnorm`, `mia_reference`, `exact_memorization` and `forget_Q_A_gibberish`.
- `setup_data.py --eval_logs` also brings in published logs for `retain90`, `retain95` and `retain99` models across Llama-3.2-1B, 3B, Llama-3.1-8B and Llama-2-7B. `forget_quality` is always computed relative to one of these, which is also how you'd evaluate your own unlearned model.

## Repository layout

```
scripts/
  install.sh                  install into its own venv on aarch64 + Blackwell
  run_eval.sh                 the working evaluation command
  compare_to_published.py     my TOFU_SUMMARY.json vs the published one
results/tofu_Llama-3.2-1B-Instruct_full/
  TOFU_SUMMARY.json           aggregated metrics
  TOFU_EVAL.json              per-sample output
  eval.log                    OpenUnlearning's own log
  hydra/                      resolved Hydra config and overrides for the run
logs/
  install.log
  eval.log                    full console output of run_eval.sh
env/
  pip-freeze.txt              the venv after installation
```

The run was originally saved under a different task name, which I renamed to `tofu_full_eval` in these files. Apart from that and my home directory paths, the results and logs are unedited.

## Environment

- NVIDIA DGX Spark: GB10 Grace Blackwell GPU, aarch64, about 119 GiB unified CPU/GPU memory, Ubuntu 24.04
- Python 3.12.3, PyTorch 2.14.0+cu130, transformers 4.51.3, deepspeed 0.15.4
- Full list in `env/pip-freeze.txt`

## References

[1] V. Dorna, A. Mekala, W. Zhao, A. McCallum, Z. C. Lipton, J. Z. Kolter, P. Maini. *OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics.* arXiv:2506.12618, 2025. https://arxiv.org/abs/2506.12618

[2] P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, J. Z. Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* First Conference on Language Modeling (COLM), 2024.

```bibtex
@article{openunlearning2025,
  title   = {{OpenUnlearning}: Accelerating {LLM} Unlearning via Unified Benchmarking of Methods and Metrics},
  author  = {Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Lipton, Zachary C and Kolter, J Zico and Maini, Pratyush},
  journal = {arXiv preprint arXiv:2506.12618},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.12618}
}

@inproceedings{maini2024tofu,
  title     = {{TOFU}: A Task of Fictitious Unlearning for {LLMs}},
  author    = {Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle = {First Conference on Language Modeling},
  year      = {2024}
}
```

## License

The scripts and notes in this repo are MIT licensed. OpenUnlearning is MIT licensed by CMU Locus Lab, and the evaluation outputs here were produced with it and their released checkpoint.
