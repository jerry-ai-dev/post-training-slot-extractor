from __future__ import annotations

import re
from collections import Counter

from slot_extractor.schemas.sample import ReplyExpectations

_REPLACEMENTS = (
    ("一个半小时", "90分钟"),
    ("一個半小時", "90分钟"),
    ("1.5小时", "90分钟"),
    ("一个小时", "60分钟"),
    ("一個小時", "60分钟"),
    ("一小时", "60分钟"),
    ("1小时", "60分钟"),
    ("半个小时", "30分钟"),
    ("半個小時", "30分钟"),
    ("半小时", "30分钟"),
    ("两点", "2点"),
    ("二点", "2点"),
    ("三点", "3点"),
    ("四点", "4点"),
    ("五点", "5点"),
    ("六点", "6点"),
    ("七点", "7点"),
    ("八点", "8点"),
    ("九点", "9点"),
    ("十点", "10点"),
    ("十一点", "11点"),
    ("十二点", "12点"),
)

_SPEECH_ACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "ask_for_start_time": (
        r"什么时候",
        r"什么时间",
        r"什么具体时间",
        r"几点",
        r"哪天",
        r"预约时间",
        r"具体时间.{0,8}(?:说|告知|提供|确定|方便)",
    ),
    "ask_for_duration": (
        r"多久",
        r"多长时间",
        r"多少分钟",
        r"几分钟",
        r"服务时长",
        r"时长.{0,8}(?:说|告知|提供|确定|方便)",
    ),
    "inform_technician_available": (
        r"技师.{0,8}有空",
        r"技师.{0,8}(?<!不)可用",
        r"技师.{0,8}(?<!不)可以服务",
        r"(?<!不)可以.{0,8}安排.{0,8}技师",
        r"技师.{0,8}(?<!不)可以为您",
        r"(?<!没)有空",
        r"(?<!不)可以服务",
        r"技师.{0,24}(?<!不)可.{0,24}提供",
        r"(?<!不)可.{0,24}提供",
        r"(?<!不)可以.{0,24}提供",
        r"(?<!不)可以.{0,8}安排",
        r"已找到.{0,12}技师",
    ),
    "request_confirmation": (
        r"确认",
        r"您看.{0,8}可以吗",
        r"这个安排.{0,8}可以吗",
        r"需要.{0,12}安排吗",
        r"是否.{0,12}安排",
        r"您觉得.{0,8}可以吗",
    ),
    "inform_technician_unavailable": (
        r"技师.{0,8}没空",
        r"技师.{0,8}没有空",
        r"技师.{0,8}不可用",
        r"技师.{0,8}排满",
        r"技师.{0,12}无法.{0,8}提供",
        r"技师.{0,12}不能.{0,8}提供",
        r"无法.{0,12}提供.{0,8}服务",
    ),
    "inform_technician_not_found": (
        r"没有找到.{0,8}技师",
        r"找不到.{0,8}技师",
        r"技师.{0,8}不存在",
        r"暂无.{0,8}技师",
        r"未能找到.{0,8}技师",
        r"未找到.{0,12}技师",
    ),
    "inform_no_match": (
        r"没有.{0,12}符合.{0,12}技师",
        r"没有匹配.{0,8}技师",
        r"暂无.{0,12}合适.{0,8}技师",
        r"未找到.{0,12}符合",
        r"没有找到.{0,12}技师",
    ),
    "acknowledge_booking_authorization": (
        r"正在.{0,8}办理预约",
        r"将为您.{0,8}办理",
        r"为您处理预约",
        r"授权.{0,8}创建预约",
        r"确认.{0,16}提交预约",
        r"提交.{0,8}预约",
        r"上层.{0,8}创建预约",
    ),
    "acknowledge_result": (
        r"好的",
        r"了解",
        r"知道了",
        r"如需.{0,12}告诉我",
    ),
    "acknowledge_pause": (
        r"暂时.{0,6}不.{0,6}预约",
        r"暂不.{0,6}预约",
        r"先不.{0,6}预约",
        r"需要时.{0,8}告诉我",
        r"暂停预约",
    ),
    "claim_booking_success": (
        r"预约成功",
        r"预约好了",
        r"已经.{0,8}预约",
        r"已为您.{0,8}安排好",
        r"已经.{0,8}安排好",
    ),
}


def normalize_reply(text: str) -> str:
    normalized = text.strip().lower()
    for source, target in _REPLACEMENTS:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()【】\[\]—-]+", "", normalized)
    return normalized


def has_speech_act(text: str, act: str) -> bool:
    patterns = _SPEECH_ACT_PATTERNS.get(act)
    if patterns is None:
        raise ValueError(f"unsupported reply speech act: {act}")
    normalized = normalize_reply(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def _ngrams(text: str, size: int) -> Counter[str]:
    if not text:
        return Counter()
    if len(text) < size:
        return Counter({text: 1})
    return Counter(text[index : index + size] for index in range(len(text) - size + 1))


def _counter_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _reference_similarity(text: str, reference: str) -> float:
    normalized_text = normalize_reply(text)
    normalized_reference = normalize_reply(reference)
    scores = [
        _counter_f1(_ngrams(normalized_text, size), _ngrams(normalized_reference, size))
        for size in (2, 3)
    ]
    return sum(scores) / len(scores)


def max_reference_similarity(text: str, references: tuple[str, ...]) -> float:
    if not references:
        return 1.0
    return max(_reference_similarity(text, reference) for reference in references)


def semantic_reply_score(
    text: str,
    expectations: ReplyExpectations,
) -> tuple[float, bool, str]:
    required_hits = [act for act in expectations.required_acts if has_speech_act(text, act)]
    forbidden_hits = [act for act in expectations.forbidden_acts if has_speech_act(text, act)]
    required_total = len(expectations.required_acts)
    coverage = len(required_hits) / required_total if required_total else 1.0
    similarity = max_reference_similarity(text, expectations.references)
    score = 0.70 * coverage + 0.30 * similarity
    passed = coverage == 1.0 and not forbidden_hits
    detail = (
        f"required={len(required_hits)}/{required_total}; "
        f"forbidden={','.join(forbidden_hits) if forbidden_hits else 'none'}; "
        f"reference_similarity={similarity:.3f}"
    )
    return score, passed, detail
