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
from slot_extractor.data.phase06_sft import _final, _input, _sample, _state, _tool
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.sft_render import render_sft
from slot_extractor.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Round3BuildResult:
    raw_path: Path
    dataset_info_path: Path
    manifest_path: Path
    model_splits: dict[str, tuple[Path, Path]]


def _history_with_tool_result(
    *,
    call_id: str,
    start: str,
    duration: int,
    preferences: list[str],
    technician_name: str | None,
    status: str,
    result_name: str | None,
    result_gender: str | None,
) -> list[dict[str, Any]]:
    arguments = json.dumps(
        {
            "technician_name": technician_name,
            "start_time": start,
            "duration_minutes": duration,
            "gender_preference": None,
            "preferences": preferences,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload: dict[str, Any] = {"mode": "specific", "status": status}
    if technician_name is not None:
        payload["requested_technician"] = technician_name
    if result_name is not None:
        payload["technician"] = {"name": result_name, "gender": result_gender}
    return [
        {"role": "user", "content": "请按刚才给出的完整条件查询。"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "find_technicians", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def generate_shared_round3_samples() -> list[RawSample]:
    """Cross-model fixes for semantic slots, date updates, and result preservation."""
    samples: list[RawSample] = []
    anchor = datetime(2028, 1, 8, 9, 0)

    # Generic service words, durations, and gender requests must not leak into preferences.
    for index in range(30):
        name = NAMES[index % len(NAMES)]
        duration = DURATIONS[index % len(DURATIONS)]
        target = (anchor + timedelta(days=index + 3)).replace(hour=(10, 14, 16)[index % 3])
        start = target.strftime("%Y-%m-%d %H:%M")
        mode = index % 3
        if mode == 0:
            user = f"{target.month}月{target.day}日{target.hour}点找{name}按摩{duration}分钟"
            expected = _tool(start, duration, [], technician_name=name)
            tags = ["偏好排除", "按摩不是偏好"]
        elif mode == 1:
            user = f"{target.month}月{target.day}日{target.hour}点按摩{duration}分钟，想要女技师"
            expected = _tool(start, duration, [], gender_preference="female")
            tags = ["偏好排除", "性别不是偏好"]
        else:
            preference = PREFERENCES[index % len(PREFERENCES)]
            user = f"{target.month}月{target.day}日{target.hour}点做{duration}分钟{preference}按摩"
            expected = _tool(start, duration, [preference])
            tags = ["偏好保留", "时长不是偏好"]
        samples.append(
            _sample(
                f"phase06-r3-slot-{index + 1:03d}",
                "工具调用",
                [*tags, "字段边界"],
                _input(user_input=user, current_time=anchor + timedelta(minutes=index)),
                expected,
            )
        )

    # Relative dates are resolved from current_time while untouched state is preserved exactly.
    for index in range(40):
        name = NAMES[index % len(NAMES)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        duration = DURATIONS[index % len(DURATIONS)]
        current = anchor + timedelta(days=50 + index)
        old_dt = current.replace(hour=14, minute=0)
        old_start = old_dt.strftime("%Y-%m-%d %H:%M")
        state = _state(old_start, duration, [preference], technician_name=name)
        delta = 2 if index % 2 == 0 else 1
        hour = 16 if index % 2 == 0 else 19
        phrase = "后天下午四点" if delta == 2 else "明天晚上七点"
        target = (current + timedelta(days=delta)).replace(hour=hour, minute=0)
        if index % 4 < 2:
            user = f"只改到{phrase}，技师、时长和项目都不变"
            expected = _tool(
                target.strftime("%Y-%m-%d %H:%M"), duration, [preference], technician_name=name
            )
            tags = ["只改时间"]
        else:
            new_preference = PREFERENCES[(index + 2) % len(PREFERENCES)]
            user = f"改成{phrase}并换成{new_preference}，其他保持原样"
            expected = _tool(
                target.strftime("%Y-%m-%d %H:%M"), duration, [new_preference], technician_name=name
            )
            tags = ["时间偏好组合修改"]
        history = [
            {"role": "user", "content": "先记录这个预约方案。"},
            {"role": "assistant", "content": "方案已记录，可以继续修改。"},
        ]
        samples.append(
            _sample(
                f"phase06-r3-state-{index + 1:03d}",
                "工具调用",
                ["多轮", "相对日期", "最小状态更新", *tags],
                _input(user_input=user, current_time=current, history=history, current_state=state),
                expected,
            )
        )

    # Preserve name/gender/time/duration/preferences from tool results in strict final JSON.
    for index in range(30):
        name = NAMES[index % len(NAMES)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        duration = DURATIONS[index % len(DURATIONS)]
        target = (anchor + timedelta(days=110 + index)).replace(hour=15, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        status = ("available", "unavailable", "not_found")[index % 3]
        result_name = name if status == "available" else None
        gender = "female" if index % 2 == 0 else "male"
        final_name = name if status in {"available", "unavailable"} else None
        reply_type = {
            "available": "confirm_available",
            "unavailable": "inform_unavailable",
            "not_found": "inform_not_found",
        }[status]
        reply = {
            "available": f"{name}技师该时段有空，可以安排{duration}分钟{preference}，请确认。",
            "unavailable": f"{name}技师该时段没有空，请调整条件。",
            "not_found": f"没有找到您指定的{name}技师。",
        }[status]
        samples.append(
            _sample(
                f"phase06-r3-result-{index + 1:03d}",
                "最终 JSON",
                ["工具结果映射", "字段保真", "严格JSON", f"状态-{status}"],
                _input(
                    user_input=None,
                    current_time=anchor,
                    history=_history_with_tool_result(
                        call_id=f"r3-result-{index + 1}",
                        start=start,
                        duration=duration,
                        preferences=[preference],
                        technician_name=name,
                        status=status,
                        result_name=result_name,
                        result_gender=gender if result_name else None,
                    ),
                ),
                _final(
                    start_time=start,
                    duration=duration,
                    preferences=[preference],
                    technician_name=final_name,
                    technician_status=status,
                    technician_gender=gender if status == "available" else None,
                    reply_type=reply_type,
                    reply=reply,
                ),
            )
        )

    # Regression guards for concise missing-information replies and safe acknowledgements.
    for index in range(20):
        duration = DURATIONS[index % len(DURATIONS)]
        history = None
        current_state = None
        if index % 2 == 0:
            user = f"先做{duration}分钟按摩，日期和开始时间还没想好"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[],
                reply_type="ask_start_time",
                reply="请问您想预约哪一天、几点开始？",
            )
            tags = ["缺少时间", "回复语义回放"]
        else:
            name = NAMES[index % len(NAMES)]
            preference = PREFERENCES[index % len(PREFERENCES)]
            start = (
                (anchor + timedelta(days=index + 170))
                .replace(hour=15, minute=0)
                .strftime("%Y-%m-%d %H:%M")
            )
            user = "我知道了，先不继续预约"
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="unavailable",
                reply_type="appointment_paused",
                reply="好的，已暂停本次预约。",
            )
            current_state = _state(
                start,
                duration,
                [preference],
                technician_name=name,
                status="unavailable",
            )
            history = [
                {"role": "user", "content": "请查询刚才的预约方案。"},
                {"role": "assistant", "content": f"{name}技师该时段没有空。"},
            ]
            tags = ["暂停预约", "动作边界"]
        samples.append(
            _sample(
                f"phase06-r3-guard-{index + 1:03d}",
                "追问" if index % 2 == 0 else "最终 JSON",
                ["回归保护", *tags],
                _input(
                    user_input=user,
                    current_time=anchor + timedelta(minutes=100 + index),
                    history=history,
                    current_state=current_state,
                ),
                expected,
            )
        )
    return samples


def generate_small_round3_specialty() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2028, 7, 1, 8, 0)
    for index in range(60):
        name = NAMES[index % len(NAMES)]
        preference = [] if index % 3 == 0 else [PREFERENCES[index % len(PREFERENCES)]]
        duration = DURATIONS[index % len(DURATIONS)]
        current = anchor + timedelta(days=index)
        old = current.replace(hour=10, minute=0)
        target = (current + timedelta(days=2)).replace(hour=16, minute=0)
        state = _state(old.strftime("%Y-%m-%d %H:%M"), duration, preference, technician_name=name)
        samples.append(
            _sample(
                f"phase06-r3-small-{index + 1:03d}",
                "工具调用",
                ["0.6B专项", "短JSON", "相对日期", "参数完整性"],
                _input(
                    user_input="改到后天下午四点重新查询，其他条件全部保留",
                    current_time=current,
                    history=[
                        {"role": "user", "content": "先按原方案查询。"},
                        {"role": "assistant", "content": "原方案没有合适结果。"},
                    ],
                    current_state=state,
                ),
                _tool(
                    target.strftime("%Y-%m-%d %H:%M"), duration, preference, technician_name=name
                ),
            )
        )
    return samples


def generate_large_round3_specialty() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2028, 10, 1, 10, 0)
    for index in range(60):
        mode = index % 4
        name = NAMES[index % len(NAMES)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        duration = DURATIONS[index % len(DURATIONS)]
        target = (anchor + timedelta(days=index + 2)).replace(hour=14, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        history = None
        if mode == 0:
            expected = _final(
                start_time=None,
                duration=None,
                preferences=[],
                reply_type="handoff",
                reply=None,
            )
            expected["unrelated"] = True
            expected["missing_info"] = []
            user = "帮我写一段旅游景点介绍，这和预约无关"
            tags = ["无关请求", "handoff"]
        elif mode == 1:
            expected = _final(
                start_time=start,
                duration=None,
                preferences=[preference],
                technician_name=name,
                reply_type="ask_duration",
                reply="时间、项目和技师已记录，请问需要多长时间？",
            )
            user = f"{target.month}月{target.day}日下午两点找{name}做{preference}，时长没定"
            tags = ["缺少时长", "禁止提前调用"]
        else:
            status = "available" if mode == 2 else "not_found"
            result_name = name if status == "available" else None
            history = _history_with_tool_result(
                call_id=f"r3-large-{index + 1}",
                start=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                status=status,
                result_name=result_name,
                result_gender="female" if result_name else None,
            )
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name if status == "available" else None,
                technician_status=status,
                technician_gender="female" if status == "available" else None,
                reply_type="confirm_available" if status == "available" else "inform_not_found",
                reply=(
                    f"{name}技师该时段有空，可以安排{duration}分钟{preference}，请确认。"
                    if status == "available"
                    else f"没有找到您指定的{name}技师。"
                ),
            )
            user = None
            tags = ["工具结果", "严格JSON", "字段保真"]
        samples.append(
            _sample(
                f"phase06-r3-large-{index + 1:03d}",
                "最终 JSON",
                ["1.7B专项", *tags],
                _input(
                    user_input=user, current_time=anchor + timedelta(minutes=index), history=history
                ),
                expected,
            )
        )
    return samples


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


def build_round3_datasets(
    source_raw: str | Path = "data/raw/v0.2/samples.jsonl",
    eval_path: str | Path = "data/eval/test.jsonl",
    output_root: str | Path = "data",
    *,
    version: str = "v0.4",
    seed: int = 20260821,
) -> Round3BuildResult:
    replay = [raw_sample_from_record(row) for row in read_jsonl(source_raw)]
    r2_shared = generate_round2_shared()
    r2_small = generate_round2_small()
    r2_large = generate_round2_large()
    shared = generate_shared_round3_samples()
    small = generate_small_round3_specialty()
    large = generate_large_round3_specialty()
    views = {
        "small": [*replay, *r2_shared, *r2_small, *shared, *small],
        "large": [*replay, *r2_shared, *r2_large, *shared, *large],
    }
    union = [*replay, *r2_shared, *r2_small, *r2_large, *shared, *small, *large]
    ids = [sample.id for sample in union]
    fingerprints = [input_fingerprint(sample) for sample in union]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample id in Round 003 data")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("duplicate input in Round 003 data")
    for sample in union:
        validate_raw_sample(sample)
    assert_no_eval_overlap(union, read_jsonl(eval_path))

    root = Path(output_root)
    raw_path = root / "raw" / version / "samples.jsonl"
    write_jsonl(raw_path, [asdict(sample) for sample in union])
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
            dataset_info[f"phase06_sft_{view}_{split}_v0_4"] = {
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
        "dataset_id": "sft-v0.4",
        "parent": "sft-v0.3",
        "seed": seed,
        "counts": {
            "replay_v0.2": len(replay),
            "round2_shared": len(r2_shared),
            "round2_small_specialty": len(r2_small),
            "round2_large_specialty": len(r2_large),
            "round3_shared_new": len(shared),
            "round3_small_specialty_new": len(small),
            "round3_large_specialty_new": len(large),
            "union": len(union),
            "small_view": len(views["small"]),
            "large_view": len(views["large"]),
        },
        "tags": dict(sorted(Counter(tag for sample in union for tag in sample.tags).items())),
        "eval_exact_input_overlap": 0,
        "files": {
            "raw": {"path": raw_path.as_posix(), "sha256": _sha256(raw_path)},
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
        "# Phase 06 SFT dataset v0.4\n\n"
        "Round 003 regression-controlled SFT dataset. It replays the corresponding v0.3 "
        "model view and adds 120 shared plus 60 model-specific samples.\n\n"
        f"- Small-model view: {len(views['small'])}\n"
        f"- Large-model view: {len(views['large'])}\n"
        "- Exact eval input overlap: 0\n- DPO: not used\n",
        encoding="utf-8",
    )
    return Round3BuildResult(raw_path, dataset_info_path, manifest_path, model_splits)
