from datetime import time, timedelta

from .fixture_store import FixtureStore
from .models import (
    CanonicalToolResult,
    TechnicianMatch,
    TechnicianTrace,
    ToolQuery,
)

SUPPORTED = {"肩颈", "精油", "足部", "泰式"}
ALIASES = {
    "肩颈按摩": "肩颈",
    "精油按摩": "精油",
    "足部按摩": "足部",
    "泰式按摩": "泰式",
}


class FindTechniciansExecutor:
    def __init__(self, store: FixtureStore) -> None:
        self.store = store

    def find(self, query: ToolQuery) -> CanonicalToolResult:
        end = query.start_time + timedelta(minutes=query.duration_minutes)
        dates = {
            window.start.date()
            for technician in self.store.technicians()
            for window in technician.availability
        }
        if (
            query.duration_minutes <= 0
            or query.start_time.date() != end.date()
            or query.start_time.date() not in dates
            or query.start_time.time() < time(9)
            or end.time() > time(21)
        ):
            return self._result(
                query, "mock_coverage_miss", (), (), "查询超出 Demo 日历范围", "unsupported_time"
            )
        normalized_preferences = tuple(ALIASES.get(item, item) for item in query.preferences)
        unsupported = tuple(
            original
            for original, normalized in zip(
                query.preferences, normalized_preferences, strict=True
            )
            if normalized not in SUPPORTED
        )
        if unsupported:
            return self._result(
                query,
                "mock_coverage_miss",
                (),
                (),
                f"未建模偏好：{'、'.join(unsupported)}",
                "unsupported_preferences",
            )
        technicians = self.store.technicians()
        named = next((tech for tech in technicians if tech.name == query.technician_name), None)
        if query.technician_name and named is None:
            traces = tuple(
                TechnicianTrace(tech.name, False, False, ("姓名不匹配",)) for tech in technicians
            )
            return self._result(query, "not_found", (), traces, "指定技师不存在")
        traces = []
        matches = []
        for technician in technicians:
            considered = query.technician_name is None or technician.name == query.technician_name
            reasons = []
            if not considered:
                reasons.append("姓名不匹配")
            if (
                considered
                and query.gender_preference
                and technician.gender != query.gender_preference
            ):
                reasons.append("性别不匹配")
            missing = [
                original
                for original, normalized in zip(
                    query.preferences, normalized_preferences, strict=True
                )
                if normalized not in technician.specialties
            ]
            if considered and missing:
                reasons.append(f"缺少专长：{'、'.join(missing)}")
            if considered and not any(
                window.contains(query.start_time, end) for window in technician.availability
            ):
                reasons.append("时间不可用")
            matched = considered and not reasons
            traces.append(TechnicianTrace(technician.name, considered, matched, tuple(reasons)))
            if matched:
                matches.append(TechnicianMatch(technician.name, technician.gender))
        if query.technician_name:
            status = "available" if matches else "unavailable"
        elif len(matches) == 1:
            status = "matched"
        elif len(matches) == 0:
            status = "no_match"
        else:
            return self._result(
                query,
                "mock_coverage_miss",
                tuple(matches),
                tuple(traces),
                "查询命中多个候选，Demo 不静默选择",
                "ambiguous_candidates",
            )
        return self._result(query, status, tuple(matches), tuple(traces), f"匹配结果：{status}")

    @staticmethod
    def _result(query, status, candidates, trace, explanation, error_code=None):
        return CanonicalToolResult(status, query, candidates, trace, explanation, error_code)
