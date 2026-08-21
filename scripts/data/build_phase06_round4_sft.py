from __future__ import annotations

from slot_extractor.data.phase06_round4_sft import build_round4_datasets


def main() -> int:
    result = build_round4_datasets()
    print(result.manifest_path)
    print(result.holdout_path)
    for view, paths in result.model_splits.items():
        print(view, *paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
