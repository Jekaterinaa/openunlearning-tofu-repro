#!/usr/bin/env python3
"""Compare my TOFU_SUMMARY.json with the one OpenUnlearning publishes for the same model.

`python setup_data.py --eval_logs` downloads the authors' own evaluation logs, including
saves/eval/tofu_Llama-3.2-1B-Instruct_full/TOFU_SUMMARY.json, so a local run can be checked
against their numbers directly.

Usage (from the open-unlearning checkout):
    python compare_to_published.py \
        saves/eval/tofu_Llama-3.2-1B-Instruct_full/TOFU_SUMMARY.json \
        saves/eval/tofu_full_eval/TOFU_SUMMARY.json
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    published, mine = (json.load(open(p)) for p in sys.argv[1:])
    print(f"| Metric | Published | Mine | Difference |\n|---|---|---|---|")
    for key in sorted(set(published) | set(mine)):
        p, m = published.get(key), mine.get(key)
        diff = f"{m - p:+.4g}" if isinstance(p, (int, float)) and isinstance(m, (int, float)) else ""
        print(f"| `{key}` | {p} | {m} | {diff} |")


if __name__ == "__main__":
    main()
