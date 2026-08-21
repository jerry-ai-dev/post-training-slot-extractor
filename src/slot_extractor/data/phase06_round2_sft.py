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
from slot_extractor.data.phase06_sft import _final, _input, _sample, _state, _tool
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.sft_render import render_sft
from slot_extractor.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Round2BuildResult:
    raw_path: Path
    dataset_info_path: Path
    manifest_path: Path
    model_splits: dict[str, tuple[Path, Path]]


NAMES = ("方宁", "高远", "林薇", "许晨", "蒋欣", "陆航", "唐悦", "韩松")
PREFERENCES = ("肩背", "足疗", "精油放松", "中式", "轻柔力度", "安静环境")
DURATIONS = (45, 60, 90, 120)


def _history(
    state: dict[str, Any], *, assistant: str = "当前方案已记录，请确认或修改。"
) -> list[dict[str, Any]]:
    del state
    return [
        {"role": "user", "content": "我先说一下预约方案。"},
        {"role": "assistant", "content": assistant},
    ]


def generate_shared_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2027, 2, 3, 9, 0)
    index = 1
    # Dense minimal pairs: missing time, missing duration, complete, and one-field edits.
    for cycle in range(30):
        name = NAMES[cycle % len(NAMES)]
        preference = PREFERENCES[cycle % len(PREFERENCES)]
        duration = DURATIONS[cycle % len(DURATIONS)]
        target = (anchor + timedelta(days=cycle + 2)).replace(
            hour=(10, 14, 16, 19)[cycle % 4], minute=0
        )
        start = target.strftime("%Y-%m-%d %H:%M")
        variants = (
            (
                f"{target.month}月{target.day}日{target.hour}点找{name}做{preference}，时长还没定",
                _final(
                    start_time=start,
                    duration=None,
                    preferences=[preference],
                    technician_name=name,
                    reply_type="ask_duration",
                    reply="请问您希望服务多长时间？",
                ),
                ["缺少时长", "近邻决策"],
            ),
            (
                f"找{name}做{duration}分钟{preference}，具体时间还没定",
                _final(
                    start_time=None,
                    duration=duration,
                    preferences=[preference],
                    technician_name=name,
                    reply_type="ask_start_time",
                    reply="请告诉我具体预约日期和开始时间。",
                ),
                ["缺少时间", "近邻决策"],
            ),
            (
                f"{target.month}月{target.day}日{target.hour}点找{name}做{duration}分钟{preference}",
                _tool(start, duration, [preference], technician_name=name),
                ["信息齐全", "近邻决策"],
            ),
        )
        for user, expected, tags in variants:
            samples.append(
                _sample(
                    f"phase06-r2-boundary-{index:03d}",
                    "工具调用" if expected["action"] == "tool_call" else "追问",
                    [*tags, "动作边界"],
                    _input(user_input=user, current_time=anchor + timedelta(days=cycle)),
                    expected,
                )
            )
            index += 1

    # Multi-turn minimal updates, including relative-date changes.
    for cycle in range(30):
        name = NAMES[cycle % len(NAMES)]
        replacement = NAMES[(cycle + 3) % len(NAMES)]
        old_pref = PREFERENCES[cycle % len(PREFERENCES)]
        new_pref = PREFERENCES[(cycle + 2) % len(PREFERENCES)]
        old_duration = DURATIONS[cycle % len(DURATIONS)]
        old_dt = (anchor + timedelta(days=cycle + 8)).replace(hour=14, minute=0)
        old_start = old_dt.strftime("%Y-%m-%d %H:%M")
        state = _state(old_start, old_duration, [old_pref], technician_name=name)
        operation = cycle % 5
        if operation == 0:
            new_dt = old_dt + timedelta(days=2, hours=2)
            user = "改到后天下午四点，其他都不变"
            expected = _tool(
                new_dt.strftime("%Y-%m-%d %H:%M"), old_duration, [old_pref], technician_name=name
            )
            tags = ["相对日期", "只改时间"]
        elif operation == 1:
            user = f"只把技师换成{replacement}，其他不动"
            expected = _tool(old_start, old_duration, [old_pref], technician_name=replacement)
            tags = ["只改技师"]
        elif operation == 2:
            user = f"只改成{new_pref}，时间、时长和技师保持原样"
            expected = _tool(old_start, old_duration, [new_pref], technician_name=name)
            tags = ["只改偏好"]
        elif operation == 3:
            new_duration = DURATIONS[(cycle + 1) % len(DURATIONS)]
            user = f"只把时长改为{new_duration}分钟"
            expected = _tool(old_start, new_duration, [old_pref], technician_name=name)
            tags = ["只改时长"]
        else:
            new_dt = old_dt + timedelta(days=1, hours=3)
            user = f"明天下午五点，项目换成{new_pref}，其他不变"
            expected = _tool(
                new_dt.strftime("%Y-%m-%d %H:%M"), old_duration, [new_pref], technician_name=name
            )
            tags = ["组合修改", "相对日期"]
        for wording in (user, f"麻烦重新查一下：{user}"):
            samples.append(
                _sample(
                    f"phase06-r2-state-{index:03d}",
                    "工具调用",
                    ["多轮", "最小状态更新", *tags],
                    _input(
                        user_input=wording,
                        current_time=old_dt.replace(hour=9),
                        history=_history(state),
                        current_state=state,
                    ),
                    expected,
                )
            )
            index += 1

    # Tool-result to final-state mappings. Tool status and output status intentionally differ.
    mappings = (
        ("matched", "available", "confirm_available", "已找到合适技师，请确认这个安排。"),
        ("unavailable", "unavailable", "inform_unavailable", "指定技师该时段没有空。"),
        ("not_found", "not_found", "inform_not_found", "没有找到您指定的技师。"),
        ("no_match", "no_match", "inform_no_match", "暂时没有符合条件的技师。"),
    )
    for cycle in range(15):
        name = NAMES[cycle % len(NAMES)]
        preference = PREFERENCES[cycle % len(PREFERENCES)]
        duration = DURATIONS[cycle % len(DURATIONS)]
        target = (anchor + timedelta(days=cycle + 50)).replace(hour=15, minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        for tool_status, final_status, reply_type, reply in mappings:
            selected_name = name if tool_status in {"matched", "unavailable"} else None
            tool_payload = {
                "mode": "search",
                "status": tool_status,
                "candidates": [{"name": name, "gender": "female"}]
                if tool_status == "matched"
                else [],
            }
            tool_arguments = json.dumps(
                {
                    "technician_name": name,
                    "start_time": start,
                    "duration_minutes": duration,
                    "gender_preference": None,
                    "preferences": [preference],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            history = [
                {
                    "role": "user",
                    "content": f"查询{start}找{name}做{duration}分钟{preference}",
                },
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"r2-{cycle}-{tool_status}",
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": tool_arguments,
                    },
                }]},
                {
                    "role": "tool",
                    "tool_call_id": f"r2-{cycle}-{tool_status}",
                    "content": json.dumps(tool_payload, ensure_ascii=False),
                },
            ]
            samples.append(
                _sample(
                    f"phase06-r2-mapping-{index:03d}",
                    "最终 JSON",
                    ["工具结果映射", f"工具状态-{tool_status}", "合法枚举"],
                    _input(user_input=None, current_time=anchor, history=history),
                    _final(
                        start_time=start,
                        duration=duration,
                        preferences=[preference],
                        technician_name=selected_name,
                        technician_status=final_status,
                        reply_type=reply_type,
                        reply=reply,
                    ),
                )
            )
            index += 1
    return samples


def generate_small_model_specialty() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2027, 5, 10, 8, 0)
    for index in range(1, 81):
        name = NAMES[index % len(NAMES)] if index % 3 else None
        preference = [] if index % 4 == 0 else [PREFERENCES[index % len(PREFERENCES)]]
        duration = DURATIONS[index % len(DURATIONS)]
        target = (anchor + timedelta(days=index)).replace(hour=(10, 14, 17, 20)[index % 4])
        state = _state(
            target.strftime("%Y-%m-%d %H:%M"), duration, preference,
            technician_name=name, status="unavailable" if index % 2 else "no_match",
        )
        samples.append(
            _sample(
                f"phase06-r2-small-tool-{index:03d}",
                "工具调用",
                ["0.6B专项", "严格短JSON", "重试重选"],
                _input(
                    user_input=(
                        f"换成{NAMES[(index + 1) % len(NAMES)]}再查一次"
                        if name
                        else "条件不变，再查一次"
                    ),
                    current_time=anchor + timedelta(minutes=index),
                    history=_history(state, assistant="刚才的查询没有合适结果，您可以调整条件。"),
                    current_state=state,
                ),
                _tool(
                    target.strftime("%Y-%m-%d %H:%M"), duration, preference,
                    technician_name=NAMES[(index + 1) % len(NAMES)] if name else None,
                ),
            )
        )
    return samples


def generate_large_model_specialty() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2027, 8, 1, 10, 0)
    for index in range(1, 81):
        name = NAMES[index % len(NAMES)]
        preference = PREFERENCES[index % len(PREFERENCES)]
        duration = DURATIONS[index % len(DURATIONS)]
        target = (anchor + timedelta(days=index)).replace(hour=16, minute=0)
        mode = index % 4
        history: list[dict[str, Any]] | None = None
        if mode == 0:
            expected = _final(
                start_time=target.strftime("%Y-%m-%d %H:%M"), duration=None,
                preferences=[preference], technician_name=name, reply_type="ask_duration",
                reply="时间和技师已记录，请问需要多长时间？",
            )
            user = f"{target.month}月{target.day}日下午四点找{name}做{preference}，时长没定"
            tags = ["缺少时长", "禁止工具调用"]
        elif mode == 1:
            expected = _final(
                start_time=None, duration=duration, preferences=[preference], technician_name=name,
                reply_type="ask_start_time", reply="技师和时长已记录，请提供具体预约时间。",
            )
            user = f"找{name}做{duration}分钟{preference}，时间之后再说"
            tags = ["缺少时间", "禁止工具调用"]
        elif mode == 2:
            expected = _final(
                start_time=None, duration=None, preferences=[], reply_type="handoff",
                reply=None,
            )
            expected["unrelated"] = True
            expected["missing_info"] = []
            expected["info_complete"] = False
            user = f"请介绍一下{target.month}月{target.day}日的节气和天气常识"
            tags = ["无关请求", "动作稳定"]
        else:
            expected = _final(
                start_time=target.strftime("%Y-%m-%d %H:%M"), duration=duration,
                preferences=[preference], technician_name=name, technician_status="available",
                reply_type="confirm_available", reply=f"{name}技师该时段有空，请确认是否预约。",
            )
            call_id = f"r2-large-matched-{index}"
            arguments = json.dumps(
                {
                    "technician_name": None,
                    "start_time": target.strftime("%Y-%m-%d %H:%M"),
                    "duration_minutes": duration,
                    "gender_preference": None,
                    "preferences": [preference],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            history = [
                {
                    "role": "user",
                    "content": (
                        f"查询{target.month}月{target.day}日下午四点"
                        f"做{duration}分钟{preference}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "find_technicians",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "mode": "search",
                            "status": "matched",
                            "candidates": [{"name": name, "gender": "female"}],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            user = None
            tags = ["matched映射", "合法枚举"]
        samples.append(
            _sample(
                f"phase06-r2-large-final-{index:03d}",
                "最终 JSON",
                ["1.7B专项", *tags],
                _input(
                    user_input=user,
                    current_time=anchor + timedelta(minutes=index),
                    history=history,
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


def build_round2_datasets(
    source_raw: str | Path = "data/raw/v0.2/samples.jsonl",
    eval_path: str | Path = "data/eval/test.jsonl",
    output_root: str | Path = "data",
    *,
    version: str = "v0.3",
    seed: int = 20260821,
) -> Round2BuildResult:
    replay = [raw_sample_from_record(row) for row in read_jsonl(source_raw)]
    shared = generate_shared_samples()
    small = generate_small_model_specialty()
    large = generate_large_model_specialty()
    union = [*replay, *shared, *small, *large]
    ids = [sample.id for sample in union]
    fingerprints = [input_fingerprint(sample) for sample in union]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample id in Round 002 data")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("duplicate input in Round 002 data")
    for sample in union:
        validate_raw_sample(sample)
    assert_no_eval_overlap(union, read_jsonl(eval_path))

    root = Path(output_root)
    raw_path = root / "raw" / version / "samples.jsonl"
    write_jsonl(raw_path, [asdict(sample) for sample in union])
    views = {
        "small": [*replay, *shared, *small],
        "large": [*replay, *shared, *large],
    }
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
            dataset_info[f"phase06_sft_{view}_{split}_v0_3"] = {
                "formatting": "sharegpt",
                "tags": {
                    "role_tag": "from", "content_tag": "value", "user_tag": "human",
                    "assistant_tag": "gpt", "function_tag": "function_call",
                    "observation_tag": "observation",
                },
                "file_name": f"../sft/{version}/{view}/{split}.jsonl",
                "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            }
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = processed / "manifest.json"
    manifest = {
        "schema_version": 1,
        "dataset_id": "sft-v0.3",
        "parent": "sft-v0.2",
        "seed": seed,
        "counts": {
            "replay_v0.2": len(replay), "shared_new": len(shared),
            "small_specialty_new": len(small), "large_specialty_new": len(large),
            "union": len(union),
            "small_view": len(views["small"]), "large_view": len(views["large"]),
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
        "# Phase 06 SFT dataset v0.3\n\n"
        "Round 002 dataset with a shared residual-error core and model-specific views.\n\n"
        f"- v0.2 replay: {len(replay)}\n- shared additions: {len(shared)}\n"
        f"- 0.6B specialty: {len(small)}\n- 1.7B specialty: {len(large)}\n"
        "- Exact eval input overlap: 0\n- DPO: not used\n",
        encoding="utf-8",
    )
    return Round2BuildResult(raw_path, dataset_info_path, manifest_path, model_splits)
