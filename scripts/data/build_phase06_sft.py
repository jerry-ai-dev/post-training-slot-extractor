from __future__ import annotations

import argparse

from slot_extractor.data.phase06_sft import build_phase06_sft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 06 targeted SFT dataset")
    parser.add_argument("--source-raw", default="data/raw/v0.1/samples.jsonl")
    parser.add_argument("--eval", default="data/eval/test.jsonl")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--version", default="v0.2")
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    result = build_phase06_sft_dataset(
        args.source_raw,
        args.eval,
        args.output_root,
        version=args.version,
        seed=args.seed,
    )
    print(
        f"raw={result.raw_count}, sft_train={result.train_count}, "
        f"sft_val={result.val_count}, eval_overlap=0"
    )
    paths = (
        result.raw_path,
        result.train_path,
        result.val_path,
        result.manifest_path,
        result.card_path,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
