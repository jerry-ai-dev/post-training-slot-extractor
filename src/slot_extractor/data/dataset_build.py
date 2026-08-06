from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slot_extractor.data.coverage_audit import (
    SemanticCoverageReport,
    audit_semantic_coverage,
)
from slot_extractor.data.dpo_perturb import perturb
from slot_extractor.data.isolation import assert_no_eval_overlap
from slot_extractor.data.raw_sample import RawSample
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.sft_render import render_sft
from slot_extractor.data.tag_audit import AuditReport, audit_tags
from slot_extractor.utils.jsonl import write_jsonl


@dataclass(frozen=True)
class BuildResult:
    raw_samples: Path
    sft_train: Path
    sft_val: Path
    dpo_train: Path
    dpo_val: Path
    dataset_info: Path
    version_card: Path
    raw_count: int
    train_count: int
    val_count: int
    dpo_train_count: int
    dpo_val_count: int
    audit: AuditReport
    semantic_coverage: SemanticCoverageReport


SHAREGPT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "function_tag": "function_call",
    "observation_tag": "observation",
}


def _dataset_info(version: str) -> dict[str, Any]:
    token = version.replace(".", "_")
    common = {
        "formatting": "sharegpt",
        "tags": SHAREGPT_TAGS,
    }
    return {
        f"phase03_sft_{token}": {
            **common,
            "file_name": f"../sft/{version}/train.jsonl",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
        },
        f"phase03_sft_val_{token}": {
            **common,
            "file_name": f"../sft/{version}/val.jsonl",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
        },
        f"phase03_dpo_{token}": {
            **common,
            "file_name": f"../dpo/{version}/train.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        f"phase03_dpo_val_{token}": {
            **common,
            "file_name": f"../dpo/{version}/val.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
    }


def _split(samples: list[RawSample], seed: int) -> tuple[list[RawSample], list[RawSample]]:
    groups: dict[str, list[RawSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.category].append(sample)
    train: list[RawSample] = []
    validation: list[RawSample] = []
    val_counts = {
        category: max(1, int(len(group) * 0.10)) if len(group) > 1 else 0
        for category, group in groups.items()
    }
    target_total = max(sum(val_counts.values()), round(len(samples) * 0.10))
    remainder_order = sorted(
        groups,
        key=lambda category: (
            len(groups[category]) * 0.10 - int(len(groups[category]) * 0.10),
            category,
        ),
        reverse=True,
    )
    for category in remainder_order[: target_total - sum(val_counts.values())]:
        val_counts[category] += 1
    for category in sorted(groups):
        group = sorted(groups[category], key=lambda item: item.id)
        random.Random(f"{seed}:{category}").shuffle(group)
        val_count = val_counts[category]
        validation.extend(group[:val_count])
        train.extend(group[val_count:])
    return train, validation


def _git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _select_dpo_samples(
    train: list[RawSample],
    validation: list[RawSample],
    target_counts: dict[str, int] | None,
    seed: int,
) -> tuple[list[tuple[RawSample, str]], list[tuple[RawSample, str]]]:
    train_candidates = [(sample, target) for sample in train for target in sample.dpo_targets]
    val_candidates = [(sample, target) for sample in validation for target in sample.dpo_targets]
    if target_counts is None:
        return train_candidates, val_candidates
    selected_train: list[tuple[RawSample, str]] = []
    selected_val: list[tuple[RawSample, str]] = []
    for target, requested in target_counts.items():
        available_train = [item for item in train_candidates if item[1] == target]
        available_val = [item for item in val_candidates if item[1] == target]
        random.Random(f"{seed}:dpo:{target}:train").shuffle(available_train)
        random.Random(f"{seed}:dpo:{target}:val").shuffle(available_val)
        val_count = max(1, round(requested * 0.10)) if requested > 1 else 0
        val_take = min(val_count, len(available_val))
        train_take = min(requested - val_take, len(available_train))
        remaining = requested - train_take - val_take
        if remaining:
            extra_val = min(remaining, len(available_val) - val_take)
            val_take += extra_val
            remaining -= extra_val
        if remaining:
            extra_train = min(remaining, len(available_train) - train_take)
            train_take += extra_train
            remaining -= extra_train
        if remaining:
            raise ValueError(
                f"DPO target {target} requires {requested}, only "
                f"{len(available_train) + len(available_val)} candidates"
            )
        selected_train.extend(available_train[:train_take])
        selected_val.extend(available_val[:val_take])
    return selected_train, selected_val


def build_dataset(
    raw_samples: Iterable[RawSample],
    eval_records: Iterable[dict[str, Any]],
    output_root: str | Path,
    version: str,
    seed: int,
    thresholds: dict[str, float] | None = None,
    dpo_target_counts: dict[str, int] | None = None,
    *,
    strict_audit: bool = False,
) -> BuildResult:
    samples = list(raw_samples)
    for sample in samples:
        validate_raw_sample(sample)
    assert_no_eval_overlap(samples, eval_records)
    audit = audit_tags(samples, thresholds or {})
    semantic_coverage = audit_semantic_coverage(samples)
    if strict_audit and audit.deficits:
        raise ValueError(f"hard-case audit deficits: {sorted(audit.deficits)}")
    if strict_audit and not semantic_coverage.ok:
        raise ValueError(
            "semantic coverage deficits: "
            f"statuses={sorted(semantic_coverage.missing_statuses)}, "
            f"reply_types={sorted(semantic_coverage.missing_reply_types)}, "
            f"confirmation_false={semantic_coverage.missing_confirmation_false}, "
            "minimal_technician_replacement="
            f"{semantic_coverage.missing_minimal_technician_replacement}"
        )
    train, validation = _split(samples, seed)
    train_rows = [render_sft(sample) for sample in train]
    validation_rows = [render_sft(sample) for sample in validation]
    selected_dpo_train, selected_dpo_val = _select_dpo_samples(
        train, validation, dpo_target_counts, seed
    )
    dpo_train_rows = [perturb(sample, target) for sample, target in selected_dpo_train]
    dpo_val_rows = [perturb(sample, target) for sample, target in selected_dpo_val]

    root = Path(output_root)
    raw_path = root / "raw" / version / "samples.jsonl"
    train_path = root / "processed" / "sft" / version / "train.jsonl"
    val_path = root / "processed" / "sft" / version / "val.jsonl"
    dpo_train_path = root / "processed" / "dpo" / version / "train.jsonl"
    dpo_val_path = root / "processed" / "dpo" / version / "val.jsonl"
    info_path = root / "processed" / version / "dataset_info.json"
    card_path = root / "processed" / version / "DATASET_CARD.md"
    write_jsonl(raw_path, [asdict(sample) for sample in samples])
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, validation_rows)
    write_jsonl(dpo_train_path, dpo_train_rows)
    write_jsonl(dpo_val_path, dpo_val_rows)
    stale_pairs = root / "processed" / "dpo" / version / "pairs.jsonl"
    if stale_pairs.exists():
        stale_pairs.unlink()
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(_dataset_info(version), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    categories = Counter(sample.category for sample in samples)
    targets = Counter(target for sample in samples for target in sample.dpo_targets)
    card_path.write_text(
        "# Phase 03 dataset card\n\n"
        f"- Version: `{version}`\n"
        f"- Built (UTC): `{datetime.now(UTC).isoformat()}`\n"
        f"- Git: `{_git_revision()}`\n"
        f"- Seed: `{seed}`\n"
        f"- Raw / SFT train / SFT val / DPO train / DPO val: `{len(samples)} / "
        f"{len(train)} / {len(validation)} / {len(dpo_train_rows)} / "
        f"{len(dpo_val_rows)}`\n"
        f"- Categories: `{dict(categories)}`\n"
        f"- DPO targets: `{dict(targets)}`\n"
        "- Eval overlap: `0`\n"
        f"- Semantic coverage passed: `{semantic_coverage.ok}`\n"
        "- This smoke dataset is not a production training corpus.\n",
        encoding="utf-8",
    )
    return BuildResult(
        raw_path,
        train_path,
        val_path,
        dpo_train_path,
        dpo_val_path,
        info_path,
        card_path,
        len(samples),
        len(train),
        len(validation),
        len(dpo_train_rows),
        len(dpo_val_rows),
        audit,
        semantic_coverage,
    )
