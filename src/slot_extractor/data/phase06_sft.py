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
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.sft_render import render_sft
from slot_extractor.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Phase06SftBuildResult:
    raw_path: Path
    train_path: Path
    val_path: Path
    manifest_path: Path
    card_path: Path
    raw_count: int
    train_count: int
    val_count: int


NAMES = ("周晴", "赵磊", "孙悦", "吴桐", "郑洁", "何川")
PREFERENCES = ("肩部", "足底", "精油", "泰式", "力度轻", "安静")


def _final(
    *,
    start_time: str | None,
    duration: int | None,
    preferences: list[str],
    technician_name: str | None = None,
    technician_status: str = "not_checked",
    confirmation: bool = False,
    reply_type: str,
    reply: str,
    gender_preference: str | None = None,
    technician_gender: str | None = None,
) -> dict[str, Any]:
    missing = [
        field
        for field, value in (("start_time", start_time), ("duration_minutes", duration))
        if value is None
    ]
    return {
        "action": "final",
        "gender_preference": gender_preference,
        "technician_gender": technician_gender,
        "start_time": start_time,
        "duration_minutes": duration,
        "preferences": preferences,
        "technician_name": technician_name,
        "technician_status": technician_status,
        "confirmation": confirmation,
        "info_complete": not missing,
        "unrelated": False,
        "missing_info": missing,
        "reply_type": reply_type,
        "reply": reply,
    }


def _tool(
    start_time: str,
    duration: int,
    preferences: list[str],
    *,
    technician_name: str | None = None,
    gender_preference: str | None = None,
) -> dict[str, Any]:
    return {
        "action": "tool_call",
        "tool_name": "find_technicians",
        "arguments": {
            "technician_name": technician_name,
            "start_time": start_time,
            "duration_minutes": duration,
            "gender_preference": gender_preference,
            "preferences": preferences,
        },
    }


def _input(
    *,
    user_input: str | None,
    current_time: datetime,
    history: list[dict[str, Any]] | None = None,
    current_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "history": history or [],
        "user_input": user_input,
        "current_time": current_time.strftime("%Y-%m-%d %H:%M"),
        "current_state": current_state,
        "available_tools": ["find_technicians"],
    }


def _sample(
    sample_id: str,
    category: str,
    tags: list[str],
    input_obj: dict[str, Any],
    expected: dict[str, Any],
) -> RawSample:
    record = {
        "id": sample_id,
        "output_kind": expected["action"],
        "conversation_kind": "multi_turn" if input_obj["history"] else "single_turn",
        "tags": [category, *tags],
        "input": input_obj,
        "expected": expected,
        "dpo_targets": [],
    }
    sample = raw_sample_from_record(record)
    validate_raw_sample(sample)
    return sample


def _state(
    start_time: str,
    duration: int,
    preferences: list[str],
    *,
    technician_name: str | None = None,
    gender_preference: str | None = None,
    technician_gender: str | None = None,
    status: str = "available",
) -> dict[str, Any]:
    return {
        "start_time": start_time,
        "duration_minutes": duration,
        "preferences": preferences,
        "technician_name": technician_name,
        "technician_status": status,
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "last_reply_type": "confirm_available",
        "gender_preference": gender_preference,
        "technician_gender": technician_gender,
    }


def generate_date_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    anchors = (
        datetime(2026, 1, 30, 9, 15),
        datetime(2026, 2, 27, 16, 40),
        datetime(2026, 4, 29, 11, 5),
        datetime(2026, 8, 20, 18, 20),
        datetime(2026, 12, 30, 10, 0),
    )
    relative = (
        ("今天晚上八点", 0, 20),
        ("明天下午三点", 1, 15),
        ("后天晚上八点", 2, 20),
        ("大后天上午九点半", 3, 9),
    )
    durations = (45, 60, 90, 120)
    index = 1
    for anchor in anchors:
        for phrase, days, hour in relative:
            for variant in range(4):
                minute = 30 if "九点半" in phrase else 0
                target = (anchor + timedelta(days=days)).replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                duration = durations[variant]
                preference = PREFERENCES[(index - 1) % len(PREFERENCES)]
                gender = (None, "female", "male", None)[variant]
                gender_text = {"female": "，希望女技师", "male": "，希望男技师"}.get(
                    gender, ""
                )
                wording = (
                    f"想约{phrase}做{duration}分钟{preference}，技师不限",
                    f"帮我查{phrase}，时长{duration}分钟，偏好{preference}",
                    f"{phrase}安排{duration}分钟，想做{preference}",
                    f"预约{phrase}的{duration}分钟{preference}，不用指定技师",
                )[variant] + gender_text
                samples.append(
                    _sample(
                        f"phase06-date-{index:03d}",
                        "工具调用",
                        ["日期标准化", "跨月" if target.month != anchor.month else "相对日期"],
                        _input(user_input=wording, current_time=anchor),
                        _tool(
                            target.strftime("%Y-%m-%d %H:%M"),
                            duration,
                            [preference],
                            gender_preference=gender,
                        ),
                    )
                )
                index += 1

    # Weekday expressions use dates selected to exercise same-week and next-week arithmetic.
    weekday_cases = (
        (datetime(2026, 3, 2, 8, 0), "本周六下午四点", datetime(2026, 3, 7, 16, 0)),
        (datetime(2026, 3, 6, 13, 0), "下周一上午十点", datetime(2026, 3, 9, 10, 0)),
        (datetime(2026, 5, 31, 20, 0), "下周三下午两点", datetime(2026, 6, 3, 14, 0)),
        (datetime(2026, 12, 29, 9, 0), "下周五上午十一点", datetime(2027, 1, 8, 11, 0)),
        (datetime(2027, 1, 1, 12, 0), "本周日上午九点", datetime(2027, 1, 3, 9, 0)),
    )
    for anchor, phrase, target in weekday_cases:
        for variant in range(4):
            duration = durations[variant]
            preference = PREFERENCES[(index + variant) % len(PREFERENCES)]
            wording = (
                f"{phrase}做{duration}分钟{preference}",
                f"请查一下{phrase}有没有技师，做{duration}分钟{preference}",
                f"我想改约到{phrase}，服务{duration}分钟，偏好{preference}",
                f"预约时间定在{phrase}，{duration}分钟{preference}",
            )[variant]
            samples.append(
                _sample(
                    f"phase06-date-{index:03d}",
                    "工具调用",
                    ["日期标准化", "星期换算"],
                    _input(user_input=wording, current_time=anchor),
                    _tool(target.strftime("%Y-%m-%d %H:%M"), duration, [preference]),
                )
            )
            index += 1
    return samples


def generate_state_transition_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    base_time = datetime(2026, 9, 15, 14, 0)
    index = 1
    operations = ("time", "technician", "preference", "duration", "combined")
    for cycle in range(24):
        operation = operations[cycle % len(operations)]
        old_name = NAMES[cycle % len(NAMES)]
        new_name = NAMES[(cycle + 1) % len(NAMES)]
        old_pref = PREFERENCES[cycle % len(PREFERENCES)]
        new_pref = PREFERENCES[(cycle + 2) % len(PREFERENCES)]
        old_duration = (45, 60, 90)[cycle % 3]
        old_time = base_time + timedelta(days=cycle % 8)
        state = _state(
            old_time.strftime("%Y-%m-%d %H:%M"),
            old_duration,
            [old_pref],
            technician_name=old_name,
            technician_gender="female" if cycle % 2 == 0 else "male",
        )
        history = [
            {"role": "user", "content": f"原来想约{old_name}做{old_duration}分钟{old_pref}"},
            {"role": "assistant", "content": "当前方案已经查过，您需要确认或修改吗？"},
        ]
        name, target_time, duration, prefs, gender = (
            old_name,
            old_time,
            old_duration,
            [old_pref],
            None,
        )
        if operation == "time":
            target_time = old_time + timedelta(days=2, hours=2)
            user = "改成两天后下午四点，其他条件不变"
            tags = ["多轮", "时间修改", "最小替换"]
        elif operation == "technician":
            name = new_name
            user = f"时间和项目都不变，改查{new_name}"
            tags = ["多轮", "重选技师", "最小替换"]
        elif operation == "preference":
            prefs = [new_pref]
            user = f"改做{new_pref}，时间、时长和技师不变"
            tags = ["多轮", "偏好修改", "最小替换"]
        elif operation == "duration":
            duration = 120 if old_duration != 120 else 60
            user = f"时长改成{duration}分钟，其他不变"
            tags = ["多轮", "时长修改", "最小替换"]
        else:
            target_time = old_time + timedelta(days=1, hours=3)
            prefs = [new_pref]
            user = f"改到明天下午五点，项目换成{new_pref}，其他不变"
            tags = ["多轮", "组合修改", "最小替换"]
        # Four wording variants per state transition produce 96 cases.
        variants = (
            user,
            f"麻烦{user}",
            f"{user}，请重新查询",
            f"重新查一下：{user}",
        )
        for wording in variants:
            samples.append(
                _sample(
                    f"phase06-state-{index:03d}",
                    "工具调用",
                    tags,
                    _input(
                        user_input=wording,
                        current_time=old_time.replace(hour=10, minute=0),
                        history=history,
                        current_state=state,
                    ),
                    _tool(
                        target_time.strftime("%Y-%m-%d %H:%M"),
                        duration,
                        prefs,
                        technician_name=name,
                        gender_preference=gender,
                    ),
                )
            )
            index += 1
    return samples


def generate_missing_information_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2026, 10, 8, 9, 0)
    vague_times = ("明天下午", "后天上午", "周末晚上", "下周三下午")
    for index in range(1, 81):
        mode = index % 4
        pref = PREFERENCES[index % len(PREFERENCES)]
        duration = (45, 60, 90)[index % 3]
        if mode == 0:
            user = f"想做{duration}分钟{pref}，时间还没决定"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[pref],
                reply_type="ask_start_time",
                reply="请告诉我具体预约日期和时间。",
            )
            tags = ["缺少时间", "动作边界"]
        elif mode == 1:
            target = anchor + timedelta(days=(index % 9) + 1)
            target = target.replace(hour=15, minute=0)
            user = f"想在{target.month}月{target.day}日下午三点做{pref}，时长之后再定"
            expected = _final(
                start_time=target.strftime("%Y-%m-%d %H:%M"),
                duration=None,
                preferences=[pref],
                reply_type="ask_duration",
                reply="请问您希望服务多长时间？",
            )
            tags = ["缺少时长", "动作边界"]
        elif mode == 2:
            user = f"想预约{pref}，具体日期和时长都还没想好"
            expected = _final(
                start_time=None,
                duration=None,
                preferences=[pref],
                reply_type="ask_start_time_and_duration",
                reply="请告诉我具体预约时间和服务时长。",
            )
            tags = ["同时缺失", "动作边界"]
        else:
            vague = vague_times[index % len(vague_times)]
            user = f"{vague}做{duration}分钟{pref}，具体几点还不确定"
            expected = _final(
                start_time=None,
                duration=duration,
                preferences=[pref],
                reply_type="ask_start_time",
                reply="还需要一个具体开始时间，请问您想约几点？",
            )
            tags = ["模糊时间", "不得猜测"]
        samples.append(
            _sample(
                f"phase06-boundary-{index:03d}",
                "追问",
                tags,
                _input(user_input=user, current_time=anchor + timedelta(days=index)),
                expected,
            )
        )
    return samples


def _confirmation_history(
    name: str, start_time: str, duration: int, preference: str, gender: str
) -> list[dict[str, Any]]:
    arguments = {
        "technician_name": name,
        "start_time": start_time,
        "duration_minutes": duration,
        "gender_preference": None,
        "preferences": [preference],
    }
    call_id = f"call-{name}-{start_time.replace(' ', '-').replace(':', '')}"
    return [
        {"role": "user", "content": f"请查询{name}的档期"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {
                    "mode": "specific",
                    "status": "available",
                    "requested_technician": name,
                    "technician": {"name": name, "gender": gender},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        {"role": "assistant", "content": f"{name}该时段有空，确认预约吗？"},
    ]


def generate_confirmation_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    anchor = datetime(2026, 11, 2, 10, 0)
    accepts = ("确认预约", "可以，就这样", "好的，帮我定下来", "没问题，确认")
    rejects = ("先不约了", "暂时不要", "我再考虑一下", "取消这个方案")
    for index in range(1, 41):
        name = NAMES[index % len(NAMES)]
        gender = "female" if index % 2 == 0 else "male"
        preference = PREFERENCES[index % len(PREFERENCES)]
        duration = (45, 60, 90)[index % 3]
        target = anchor + timedelta(days=index + 1)
        target = target.replace(hour=(10, 14, 16, 19)[index % 4], minute=0)
        start = target.strftime("%Y-%m-%d %H:%M")
        state = _state(
            start,
            duration,
            [preference],
            technician_name=name,
            technician_gender=gender,
        )
        history = _confirmation_history(name, start, duration, preference, gender)
        if index <= 24:
            user = accepts[index % len(accepts)]
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="available",
                confirmation=True,
                reply_type="booking_authorized",
                reply=f"好的，{name}技师的预约已成功。",
                technician_gender=gender,
            )
            tags = ["可用方案确认", "预约成功"]
        else:
            user = rejects[index % len(rejects)]
            expected = _final(
                start_time=start,
                duration=duration,
                preferences=[preference],
                technician_name=name,
                technician_status="available",
                confirmation=False,
                reply_type="appointment_paused",
                reply="好的，本次暂不预约。",
                technician_gender=gender,
            )
            tags = ["拒绝确认", "预约暂缓"]
        samples.append(
            _sample(
                f"phase06-confirm-{index:03d}",
                "确认",
                tags,
                _input(
                    user_input=user,
                    current_time=anchor,
                    history=history,
                    current_state=state,
                ),
                expected,
            )
        )
    return samples


def generate_targeted_samples() -> list[RawSample]:
    samples = [
        *generate_date_samples(),
        *generate_state_transition_samples(),
        *generate_missing_information_samples(),
        *generate_confirmation_samples(),
    ]
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate targeted sample id")
    return samples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_phase06_sft_dataset(
    source_raw: str | Path,
    eval_path: str | Path,
    output_root: str | Path,
    *,
    version: str = "v0.2",
    seed: int = 20260820,
) -> Phase06SftBuildResult:
    original_base_samples = [raw_sample_from_record(row) for row in read_jsonl(source_raw)]
    base_samples: list[RawSample] = []
    seen_inputs: set[str] = set()
    for sample in original_base_samples:
        fingerprint = input_fingerprint(sample)
        if fingerprint not in seen_inputs:
            seen_inputs.add(fingerprint)
            base_samples.append(sample)
    targeted = generate_targeted_samples()
    samples = [*base_samples, *targeted]
    for sample in samples:
        validate_raw_sample(sample)
    assert_no_eval_overlap(samples, read_jsonl(eval_path))

    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample id across base and targeted data")

    by_category: dict[str, list[RawSample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    train: list[RawSample] = []
    validation: list[RawSample] = []
    for category, group in sorted(by_category.items()):
        ordered = sorted(group, key=lambda item: item.id)
        random.Random(f"{seed}:{category}").shuffle(ordered)
        val_count = round(len(ordered) * 0.10)
        validation.extend(ordered[:val_count])
        train.extend(ordered[val_count:])

    root = Path(output_root)
    raw_path = root / "raw" / version / "samples.jsonl"
    train_path = root / "processed" / "sft" / version / "train.jsonl"
    val_path = root / "processed" / "sft" / version / "val.jsonl"
    manifest_path = root / "processed" / version / "manifest.json"
    card_path = root / "processed" / version / "DATASET_CARD.md"
    write_jsonl(raw_path, [asdict(sample) for sample in samples])
    write_jsonl(train_path, [render_sft(sample) for sample in train])
    write_jsonl(val_path, [render_sft(sample) for sample in validation])

    categories = Counter(sample.category for sample in samples)
    targeted_tags = Counter(tag for sample in targeted for tag in sample.tags[1:])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset_id": f"sft-{version}",
        "parent": "sft-v0.1",
        "seed": seed,
        "source_raw": str(source_raw).replace("\\", "/"),
        "eval_dataset": "eval-v0.2",
        "counts": {
            "total": len(samples),
            "replayed_v0.1": len(base_samples),
            "duplicate_v0.1_removed": len(original_base_samples) - len(base_samples),
            "targeted_new": len(targeted),
            "train": len(train),
            "validation": len(validation),
        },
        "categories": dict(sorted(categories.items())),
        "targeted_tags": dict(sorted(targeted_tags.items())),
        "eval_exact_input_overlap": 0,
        "files": {},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["files"] = {
        "raw": {"path": str(raw_path).replace("\\", "/"), "sha256": _sha256(raw_path)},
        "train": {"path": str(train_path).replace("\\", "/"), "sha256": _sha256(train_path)},
        "validation": {"path": str(val_path).replace("\\", "/"), "sha256": _sha256(val_path)},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    card_path.write_text(
        "# Phase 06 SFT dataset v0.2\n\n"
        "- Parent: `sft-v0.1`\n"
        f"- Total / train / validation: `{len(samples)} / {len(train)} / {len(validation)}`\n"
        f"- Replayed unique v0.1 / removed duplicate inputs / targeted new: "
        f"`{len(base_samples)} / {len(original_base_samples) - len(base_samples)} / "
        f"{len(targeted)}`\n"
        f"- Categories: `{dict(sorted(categories.items()))}`\n"
        "- Exact eval input overlap: `0`\n"
        "- DPO data: not generated in Round 001\n\n"
        "## Targeted additions\n\n"
        "- Relative dates, weekdays, cross-month and cross-year normalization.\n"
        "- Multi-turn minimal replacement for time, technician, preference and duration.\n"
        "- Missing-information boundaries that must ask instead of calling tools.\n"
        "- Available-plan acceptance and rejection with consistent booking semantics.\n\n"
        "Date answers are computed during dataset construction only to create and validate labels. "
        "The trained model remains responsible for date understanding and normalization "
        "at inference time.\n",
        encoding="utf-8",
    )
    return Phase06SftBuildResult(
        raw_path,
        train_path,
        val_path,
        manifest_path,
        card_path,
        len(samples),
        len(train),
        len(validation),
    )
