from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from slot_extractor.data.isolation import assert_no_eval_overlap, input_fingerprint
from slot_extractor.data.phase06_round2_sft import (
    DURATIONS,
    NAMES,
    PREFERENCES,
)
from slot_extractor.data.phase06_round2_sft import (
    generate_large_model_specialty as generate_round2_large,
)
from slot_extractor.data.phase06_round2_sft import (
    generate_shared_samples as generate_round2_shared,
)
from slot_extractor.data.phase06_round2_sft import (
    generate_small_model_specialty as generate_round2_small,
)
from slot_extractor.data.phase06_round3_sft import (
    _history_with_tool_result,
    generate_large_round3_specialty,
    generate_shared_round3_samples,
)
from slot_extractor.data.phase06_sft import _final, _input, _sample, _state, _tool
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.sft_render import render_sft
from slot_extractor.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Round4BuildResult:
    manifest_path: Path
    holdout_path: Path
    model_splits: dict[str, tuple[Path, Path]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(samples: list[RawSample], seed: int) -> tuple[list[RawSample], list[RawSample]]:
    groups: dict[str, list[RawSample]] = {}
    for sample in samples:
        groups.setdefault(sample.category, []).append(sample)
    train: list[RawSample] = []
    val: list[RawSample] = []
    for category, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item.id)
        random.Random(f"{seed}:{category}").shuffle(ordered)
        count = max(1, round(len(ordered) * 0.10))
        val.extend(ordered[:count])
        train.extend(ordered[count:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(val)
    return train, val


def generate_small_round4_specialty() -> list[RawSample]:
    """Minimal 0.6B repair set; avoids replaying Round 003 final-heavy additions."""
    samples: list[RawSample] = []
    anchor = datetime(2029, 1, 5, 9, 0)
    for index in range(80):
        name = NAMES[index % len(NAMES)]
        duration = DURATIONS[index % len(DURATIONS)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        current = anchor + timedelta(days=index)
        target = (current + timedelta(days=2)).replace(hour=16, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        mode = index % 4
        if mode == 0:
            user = f"后天下午四点找{name}按摩{duration}分钟"
            expected = _tool(start, duration, [], technician_name=name)
            history = None
            state = None
            tags = ["偏好排除", "短JSON"]
        elif mode == 1:
            state = _state(
                current.replace(hour=11).strftime("%Y-%m-%d %H:%M"),
                duration,
                [preference],
                technician_name=name,
            )
            user = "只改到后天下午四点，其他全部保留"
            expected = _tool(start, duration, [preference], technician_name=name)
            history = [
                {"role": "user", "content": "记录原方案。"},
                {"role": "assistant", "content": "已记录，可以修改。"},
            ]
            tags = ["相对日期", "最小状态更新"]
        elif mode == 2:
            user = f"想做{duration}分钟按摩，具体时间还没确定"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[],
                reply_type="ask_start_time",
                reply="请问您想预约哪一天、几点开始？",
            )
            history = None
            state = None
            tags = ["缺少时间", "回归保护"]
        else:
            history = _history_with_tool_result(
                call_id=f"r4-small-{index}",
                start=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                status="available",
                result_name=name,
                result_gender="female",
            )
            user = None
            state = None
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="available",
                technician_gender="female",
                reply_type="confirm_available",
                reply=f"{name}技师该时段有空，可以安排{duration}分钟{preference}，请确认。",
            )
            tags = ["严格JSON", "工具结果字段保真"]
        samples.append(
            _sample(
                f"phase06-r4-small-{index + 1:03d}",
                "工具调用" if expected["action"] == "tool_call" else "最终 JSON",
                ["0.6B专项", *tags],
                _input(
                    user_input=user,
                    current_time=current,
                    history=history,
                    current_state=state,
                ),
                expected,
            )
        )
    return samples


def generate_large_round4_specialty() -> list[RawSample]:
    """Protect the 1.7B gains and repair its three best-baseline regressions."""
    samples: list[RawSample] = []
    anchor = datetime(2029, 5, 3, 10, 0)
    for index in range(80):
        name = NAMES[index % len(NAMES)]
        duration = DURATIONS[index % len(DURATIONS)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        target = (anchor + timedelta(days=index + 3)).replace(hour=14, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        mode = index % 4
        history: list[dict[str, Any]] | None = None
        if mode == 0:
            history = _history_with_tool_result(
                call_id=f"r4-large-confirm-{index}",
                start=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                status="available",
                result_name=name,
                result_gender="female",
            )
            history.append(
                {
                    "role": "assistant",
                    "content": f"{name}技师该时段有空，可以安排，您确认吗？",
                }
            )
            user = "确认，就按这个方案预约"
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="available",
                technician_gender="female",
                confirmation=True,
                reply_type="booking_authorized",
                reply="好的，已按该方案确认预约。",
            )
            tags = ["确认动作", "严格JSON"]
        elif mode == 1:
            history = _history_with_tool_result(
                call_id=f"r4-large-result-{index}",
                start=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                status="available",
                result_name=name,
                result_gender="female",
            )
            user = None
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="available",
                technician_gender="female",
                reply_type="confirm_available",
                reply=f"{name}技师该时段有空，可以安排{duration}分钟{preference}，请确认。",
            )
            tags = ["工具结果字段保真", "严格JSON"]
        elif mode == 2:
            user = f"找{name}做{duration}分钟按摩，时间还没确定"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[],
                technician_name=name,
                reply_type="ask_start_time",
                reply="请问您想预约哪一天、几点开始？",
            )
            tags = ["缺少时间", "偏好排除"]
        else:
            user = "请帮我查询股票行情，这不是预约需求"
            expected = _final(
                start_time=None,
                duration=None,
                preferences=[],
                reply_type="handoff",
                reply=None,
            )
            expected["unrelated"] = True
            expected["missing_info"] = []
            tags = ["无关请求", "动作边界"]
        samples.append(
            _sample(
                f"phase06-r4-large-{index + 1:03d}",
                "最终 JSON",
                ["1.7B专项", "回归保护", *tags],
                _input(
                    user_input=user, current_time=anchor + timedelta(minutes=index), history=history
                ),
                expected,
            )
        )
    return samples


def generate_round4_holdout() -> list[RawSample]:
    """A new blind set, authored before Round 004 training and never included in SFT."""
    samples: list[RawSample] = []
    anchor = datetime(2030, 2, 10, 9, 0)
    holdout_names = ("顾言", "苏禾", "程安", "叶岚")
    holdout_preferences = ("头部放松", "腰背", "芳香精油", "轻力度")
    for index in range(24):
        name = holdout_names[index % len(holdout_names)]
        preference = holdout_preferences[index % len(holdout_preferences)]
        duration = (40, 60, 80)[index % 3]
        current = anchor + timedelta(days=index)
        target = (current + timedelta(days=2)).replace(hour=16, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        mode = index % 4
        if mode == 0:
            user = f"后天下午四点找{name}按摩{duration}分钟"
            expected = _tool(start, duration, [], technician_name=name)
            history = None
            state = None
        elif mode == 1:
            user = f"找{name}做{preference}，已经定了{duration}分钟，但日期时间还没定"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                reply_type="ask_start_time",
                reply="请问您想预约哪一天、几点开始？",
            )
            history = None
            state = None
        elif mode == 2:
            state = _state(
                current.replace(hour=11).strftime("%Y-%m-%d %H:%M"),
                duration,
                [preference],
                technician_name=name,
            )
            user = "只改到后天下午四点，其余条件不变"
            expected = _tool(start, duration, [preference], technician_name=name)
            history = [
                {"role": "user", "content": "先保存这个方案。"},
                {"role": "assistant", "content": "已保存。"},
            ]
        else:
            user = "帮我推荐一本历史书，与预约无关"
            expected = _final(
                start_time=None,
                duration=None,
                preferences=[],
                reply_type="handoff",
                reply=None,
            )
            expected["unrelated"] = True
            expected["missing_info"] = []
            history = None
            state = None
        samples.append(
            _sample(
                f"phase06-r4-holdout-{index + 1:03d}",
                "工具调用"
                if expected["action"] == "tool_call"
                else ("无关" if expected.get("unrelated") else "追问"),
                ["round004盲测", "未参与训练"],
                _input(user_input=user, current_time=current, history=history, current_state=state),
                expected,
            )
        )
    return samples


def _holdout_eval_record(sample: RawSample) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": sample.id,
        "output_kind": sample.output_kind,
        "conversation_kind": sample.conversation_kind,
        "assertions": ["no_field_outside_schema"],
        "tags": list(sample.tags),
        "input": sample.input,
        "expected": sample.expected,
    }
    if sample.output_kind == "final":
        required_acts = {
            "ask_start_time": ["ask_for_start_time"],
            "handoff": [],
        }.get(sample.expected["reply_type"], [])
        reply = sample.expected.get("reply")
        record["reply_expectations"] = {
            "required_acts": required_acts,
            "forbidden_acts": ["claim_booking_success"],
            "required_fields": [],
            "references": [reply] if isinstance(reply, str) else [],
        }
    return record


def build_round4_datasets(
    source_raw: str | Path = "data/raw/v0.2/samples.jsonl",
    eval_path: str | Path = "data/eval/test.jsonl",
    output_root: str | Path = "data",
    *,
    version: str = "v0.5",
    seed: int = 20260821,
) -> Round4BuildResult:
    replay = [raw_sample_from_record(row) for row in read_jsonl(source_raw)]
    r2_shared = generate_round2_shared()
    r2_small = generate_round2_small()
    r2_large = generate_round2_large()
    r3_shared = generate_shared_round3_samples()
    r3_large = generate_large_round3_specialty()
    small_new = generate_small_round4_specialty()
    large_new = generate_large_round4_specialty()
    views = {
        "small": [*replay, *r2_shared, *r2_small, *small_new],
        "large": [*replay, *r2_shared, *r2_large, *r3_shared, *r3_large, *large_new],
    }
    union = [*views["small"], *r3_shared, *r3_large, *large_new]
    ids = [sample.id for sample in union]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample id in Round 004 data")
    if len({input_fingerprint(sample) for sample in union}) != len(union):
        raise ValueError("duplicate input in Round 004 data")
    holdout = generate_round4_holdout()
    for sample in [*union, *holdout]:
        validate_raw_sample(sample)
    frozen_eval = read_jsonl(eval_path)
    assert_no_eval_overlap(union, frozen_eval)
    assert_no_eval_overlap(holdout, frozen_eval)
    assert_no_eval_overlap(union, [asdict(sample) for sample in holdout])

    root = Path(output_root)
    raw_path = root / "raw" / version / "samples.jsonl"
    write_jsonl(raw_path, [asdict(sample) for sample in union])
    holdout_path = root / "eval" / "phase06_holdout_v0.3.jsonl"
    write_jsonl(holdout_path, [_holdout_eval_record(sample) for sample in holdout])
    model_splits: dict[str, tuple[Path, Path]] = {}
    for view, samples in views.items():
        train, val = _split(samples, seed + (0 if view == "small" else 1000))
        train_path = root / "processed" / "sft" / version / view / "train.jsonl"
        val_path = root / "processed" / "sft" / version / view / "val.jsonl"
        write_jsonl(train_path, [render_sft(sample) for sample in train])
        write_jsonl(val_path, [render_sft(sample) for sample in val])
        model_splits[view] = (train_path, val_path)

    processed = root / "processed" / version
    processed.mkdir(parents=True, exist_ok=True)
    dataset_info_path = processed / "dataset_info.json"
    dataset_info: dict[str, Any] = {}
    for view in views:
        for split in ("train", "val"):
            dataset_info[f"phase06_sft_{view}_{split}_v0_5"] = {
                "formatting": "sharegpt",
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "function_tag": "function_call",
                    "observation_tag": "observation",
                },
                "file_name": f"../sft/{version}/{view}/{split}.jsonl",
                "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            }
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = processed / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "sft-v0.5",
        "parent": "sft-v0.4",
        "seed": seed,
        "counts": {
            "small_view": len(views["small"]),
            "large_view": len(views["large"]),
            "small_new": len(small_new),
            "large_new": len(large_new),
            "blind_holdout": len(holdout),
            "union": len(union),
        },
        "tags": dict(sorted(Counter(tag for sample in union for tag in sample.tags).items())),
        "eval_exact_input_overlap": 0,
        "holdout_in_training": False,
        "files": {
            "raw": {"path": raw_path.as_posix(), "sha256": _sha256(raw_path)},
            "holdout": {"path": holdout_path.as_posix(), "sha256": _sha256(holdout_path)},
            "dataset_info": {
                "path": dataset_info_path.as_posix(),
                "sha256": _sha256(dataset_info_path),
            },
        },
    }
    for view, (train_path, val_path) in model_splits.items():
        manifest["files"][f"{view}_train"] = {
            "path": train_path.as_posix(),
            "sha256": _sha256(train_path),
        }
        manifest["files"][f"{view}_val"] = {
            "path": val_path.as_posix(),
            "sha256": _sha256(val_path),
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (processed / "DATASET_CARD.md").write_text(
        "# Phase 06 SFT dataset v0.5\n\nRound 004 final targeted SFT iteration. "
        "The 24-case blind holdout is frozen before training and excluded from both views.\n",
        encoding="utf-8",
    )
    return Round4BuildResult(manifest_path, holdout_path, model_splits)
