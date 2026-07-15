from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slot_extractor.schemas.dataset_contract import (
    DatasetContractError,
    load_dataset_contract,
    validate_dataset_against_contract,
)
from slot_extractor.schemas.sample import load_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate evaluation cases against the contract.")
    parser.add_argument("--cases", default="data/eval/test.jsonl")
    parser.add_argument("--contract", default="data/eval/dataset_contract.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_samples(Path(args.cases))
    contract = load_dataset_contract(Path(args.contract))
    try:
        validate_dataset_against_contract(samples, contract)
    except DatasetContractError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Validated {len(samples)} cases against contract v{contract['version']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
