from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from slot_extractor.data.raw_sample import RawSample

HARD_TAGS = {"相对时间", "多义短词", "多轮改口", "易混边界", "幻觉陷阱"}


@dataclass(frozen=True)
class CategoryAudit:
    total: int
    hard: int
    ratio: float


@dataclass(frozen=True)
class AuditDeficit:
    required_ratio: float
    actual_ratio: float
    missing_count: int


@dataclass(frozen=True)
class AuditReport:
    categories: dict[str, CategoryAudit]
    deficits: dict[str, AuditDeficit]


def audit_tags(samples: Iterable[RawSample], thresholds: dict[str, float]) -> AuditReport:
    counts: dict[str, list[int]] = {}
    for sample in samples:
        values = counts.setdefault(sample.category, [0, 0])
        values[0] += 1
        values[1] += int(bool(set(sample.tags) & HARD_TAGS))
    categories = {
        category: CategoryAudit(total, hard, hard / total if total else 0.0)
        for category, (total, hard) in counts.items()
    }
    deficits: dict[str, AuditDeficit] = {}
    for category, required in thresholds.items():
        audit = categories.get(category, CategoryAudit(0, 0, 0.0))
        if audit.ratio < required:
            needed = max(0, int(required * audit.total + 0.999999) - audit.hard)
            deficits[category] = AuditDeficit(required, audit.ratio, needed)
    return AuditReport(categories, deficits)
