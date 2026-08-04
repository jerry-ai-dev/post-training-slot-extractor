from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path("configs/training/llamafactory")
RENDERED_ROOT = CONFIG_ROOT / "_rendered"
STAGES = ("sft", "dpo")
OVERRIDE_KEYS = {
    "max_steps",
    "use_cpu",
    "bf16",
    "fp16",
    "output_dir",
    "overwrite_output_dir",
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def find_override(run_id: str) -> Path:
    matches = [CONFIG_ROOT / stage / f"{run_id}.yaml" for stage in STAGES]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"unknown run_id: {run_id}")
    return existing[0]


def assert_required(run_id: str, stage: str, config: Mapping[str, Any]) -> None:
    required = {"stage", "model_name_or_path", "output_dir", "dataset", "template"}
    if stage == "dpo":
        required |= {"adapter_name_or_path", "pref_beta"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{run_id} missing required keys: {', '.join(missing)}")
    if config["stage"] != stage:
        raise ValueError(f"{run_id} stage mismatch: {config['stage']!r} != {stage!r}")


def atomic_dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def render_run(
    run_id: str,
    output_root: Path = RENDERED_ROOT,
    overrides: dict[str, object] | None = None,
) -> Path:
    override_path = find_override(run_id)
    stage = override_path.parent.name
    base = load_yaml(CONFIG_ROOT / f"_base_{stage}.yaml")
    run = load_yaml(override_path)
    merged = deep_merge(base, run)
    assert_required(run_id, stage, merged)
    merged.pop("run_id", None)
    if overrides:
        unknown = sorted(set(overrides) - OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f"override key not allowed: {', '.join(unknown)}")
        merged = deep_merge(merged, overrides)
    output = output_root / f"{run_id}.yaml"
    atomic_dump_yaml(output, merged)
    return output


def parse_set_values(values: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid --set value: {value!r}")
        if key not in OVERRIDE_KEYS:
            raise ValueError(f"override key not allowed: {key}")
        result[key] = yaml.safe_load(raw)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render phase04 LLaMA-Factory configs.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--output-root", type=Path, default=RENDERED_ROOT)
    parser.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overrides = parse_set_values(args.set)
        run_ids = (
            [
                path.stem
                for stage in STAGES
                for path in sorted((CONFIG_ROOT / stage).glob("qwen3-*.yaml"))
            ]
            if args.all
            else [args.run_id]
        )
        for run_id in run_ids:
            print(render_run(run_id, output_root=args.output_root, overrides=overrides))
    except ValueError as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
