from __future__ import annotations

import argparse

from slot_extractor.data.phase06_round3_sft import build_round3_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Round 003 model-specific SFT datasets.")
    parser.add_argument("--source-raw", default="data/raw/v0.2/samples.jsonl")
    parser.add_argument("--eval", default="data/eval/test.jsonl")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    result = build_round3_datasets(args.source_raw, args.eval, args.output_root, seed=args.seed)
    print(result.manifest_path)
    for view, paths in result.model_splits.items():
        print(view, *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
