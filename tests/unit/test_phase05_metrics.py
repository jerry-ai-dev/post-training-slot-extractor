from scripts.eval.phase05_metrics import WorkloadSample, aggregate_workload, summarize


def test_aggregate_reports_required_statistics_and_separates_cold_hot():
    rows = [
        WorkloadSample("short", "cold", 100, 20, 25, 40, 65, 10, 512, 1000),
        WorkloadSample("short", "hot", 10, 20, 22, 35, 57, 10, 510, 1000),
        WorkloadSample("short", "hot", 12, 21, 23, 36, 59, 11, 511, 1000),
    ]
    result = aggregate_workload(rows)
    assert set(result.phases) == {"cold", "hot"}
    assert result.phases["hot"].total_ms == {
        "count": 2,
        "mean": 58.0,
        "median": 58.0,
        "p90": 58.8,
        "min": 57.0,
        "max": 59.0,
    }


def test_summarize_omits_missing_values_and_does_not_make_zeroes():
    assert summarize([]) == {}
    sample = WorkloadSample("short", "cold", None, None, None, None, None, None, None, None)
    assert aggregate_workload([sample]).phases["cold"].to_dict() == {}
