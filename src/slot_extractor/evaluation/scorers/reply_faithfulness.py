from __future__ import annotations

import re
from datetime import datetime, timedelta

from slot_extractor.evaluation.reply_semantics import has_speech_act, normalize_reply
from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


def _time_variants(start_time: str, current_time: str | None) -> set[str]:
    value = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    hour_text = f"{value.hour}点" if value.minute == 0 else f"{value.hour}点{value.minute}分"
    if value.hour < 12:
        period = "上午"
    elif value.hour < 18:
        period = "下午"
    else:
        period = "晚上"
    period_hour = value.hour if value.hour <= 12 else value.hour - 12
    period_text = (
        f"{period}{period_hour}点"
        if value.minute == 0
        else f"{period}{period_hour}点{value.minute}分"
    )
    variants = {
        start_time,
        value.strftime("%Y年%m月%d日%H点%M分"),
        f"{value.month}月{value.day}日{hour_text}",
        f"{value.month}月{value.day}号{hour_text}",
        value.strftime("%H:%M"),
        hour_text,
        period_text,
    }
    if current_time:
        current = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
        if value.date() == current.date():
            variants.update({f"今天{hour_text}", f"今天{period_text}"})
        elif value.date() == (current + timedelta(days=1)).date():
            variants.update({f"明天{hour_text}", f"明天{period_text}"})
        elif value.date() == (current + timedelta(days=2)).date():
            variants.update({f"后天{hour_text}", f"后天{period_text}"})
    return {normalize_reply(item) for item in variants}


def _mentioned_technicians(reply: str) -> set[str]:
    boundary = r"^|[，。！？、；：\s]|找到|安排|推荐|由"
    before_title = re.findall(
        rf"(?:{boundary})([\u4e00-\u9fff]{{2,3}})(?:老师|技师)", reply
    )
    after_title = re.findall(
        r"(?:老师|技师)([\u4e00-\u9fff]{2,3})(?=$|[，。！？、；：\s])", reply
    )
    generic_fragments = {"女", "男", "的女", "的男", "一位", "这位", "该位", "合适"}
    return {
        name
        for name in before_title + after_title
        if name not in generic_fragments and not name.endswith(("的", "位", "名", "个"))
        and not name.endswith(("女", "男"))
        and not name.endswith(("吗", "呢"))
        and not any(token in name for token in ("性别", "条件", "时间", "类型"))
    }


class ReplyFaithfulnessScorer:
    dimension = "reply_faithfulness"

    def applies_to(self, sample: Sample) -> bool:
        return sample.expected.get("action") == "final" and sample.reply_expectations is not None

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        if sample.reply_expectations is None:
            return DimensionScore(self.dimension, None, None, "no reply expectations")
        try:
            output = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        reply = output.get("reply")
        if reply is None and sample.expected.get("reply_type") == "handoff":
            return DimensionScore(self.dimension, 1.0, True, "handoff has no local reply")
        if not isinstance(reply, str) or not reply.strip():
            return DimensionScore(self.dimension, 0.0, False, "reply is missing")

        normalized = normalize_reply(reply)
        errors: list[str] = []
        checks = 2
        required_fields = sample.reply_expectations.required_fields
        expected = sample.expected

        if "technician_name" in required_fields:
            checks += 1
            name = expected.get("technician_name")
            if not isinstance(name, str) or normalize_reply(name) not in normalized:
                errors.append("technician_name")

        if "duration_minutes" in required_fields:
            checks += 1
            duration = expected.get("duration_minutes")
            duration_text = normalize_reply(f"{duration}分钟")
            if not isinstance(duration, int) or duration_text not in normalized:
                errors.append("duration_minutes")

        if "start_time" in required_fields:
            checks += 1
            start_time = expected.get("start_time")
            current_time = sample.input.get("current_time")
            if not isinstance(start_time, str) or not any(
                variant in normalized
                for variant in _time_variants(
                    start_time,
                    current_time if isinstance(current_time, str) else None,
                )
            ):
                errors.append("start_time")

        if "preferences" in required_fields:
            preferences = expected.get("preferences")
            if isinstance(preferences, list):
                checks += len(preferences)
                for preference in preferences:
                    if (
                        isinstance(preference, str)
                        and normalize_reply(preference) not in normalized
                    ):
                        errors.append(f"preference:{preference}")

        expected_name = expected.get("technician_name")
        mentioned_names = _mentioned_technicians(reply)
        if mentioned_names and (
            not isinstance(expected_name, str)
            or any(name != expected_name for name in mentioned_names)
        ):
            errors.append("unsupported_technician_name")

        status = expected.get("technician_status")
        mentions_available = has_speech_act(reply, "inform_technician_available")
        mentions_unavailable = has_speech_act(reply, "inform_technician_unavailable")
        if status == "available" and mentions_unavailable:
            errors.append("technician_status")
        if status in {"unavailable", "not_found", "no_match"} and mentions_available:
            errors.append("technician_status")

        if has_speech_act(reply, "claim_booking_success"):
            errors.append("booking_success")

        score = max(0.0, (checks - len(set(errors))) / checks)
        passed = not errors
        detail = "ok" if passed else "errors=" + ",".join(dict.fromkeys(errors))
        return DimensionScore(self.dimension, score, passed, detail)
