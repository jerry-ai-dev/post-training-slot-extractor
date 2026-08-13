"""Phase 05 workload samples and deterministic aggregation."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Literal

METRICS = (
    "load_ms", "prefill_ms", "ttft_ms", "decode_ms", "total_ms",
    "tokens", "peak_rss_mb", "file_size_bytes",
)


@dataclass(frozen=True)
class WorkloadSample:
    workload: str
    phase: Literal["cold", "hot"]
    load_ms: float | None
    prefill_ms: float | None
    ttft_ms: float | None
    decode_ms: float | None
    total_ms: float | None
    tokens: int | None
    peak_rss_mb: float | None
    file_size_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseAggregate:
    load_ms: dict[str, float | int]
    prefill_ms: dict[str, float | int]
    ttft_ms: dict[str, float | int]
    decode_ms: dict[str, float | int]
    total_ms: dict[str, float | int]
    tokens: dict[str, float | int]
    peak_rss_mb: dict[str, float | int]
    file_size_bytes: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {name: value for name, value in asdict(self).items() if value}


@dataclass(frozen=True)
class WorkloadAggregate:
    workload: str
    phases: dict[str, PhaseAggregate]


def summarize(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "mean": float(mean(ordered)),
        "median": float(median(ordered)),
        "p90": round(p90, 10),
        "min": float(min(ordered)),
        "max": float(max(ordered)),
    }


def aggregate_workload(samples: Sequence[WorkloadSample]) -> WorkloadAggregate:
    if not samples:
        raise ValueError("at least one workload sample is required")
    workload = samples[0].workload
    if any(sample.workload != workload for sample in samples):
        raise ValueError("samples must have one workload")
    phases = {}
    for phase in ("cold", "hot"):
        rows = [sample for sample in samples if sample.phase == phase]
        if rows:
            phases[phase] = PhaseAggregate(
                **{
                    metric: summarize(
                        [value for row in rows if (value := getattr(row, metric)) is not None]
                    )
                    for metric in METRICS
                }
            )
    return WorkloadAggregate(workload, phases)
