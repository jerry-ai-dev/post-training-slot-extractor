from datetime import datetime
from pathlib import Path

from slot_extractor.tool_loop.find_technicians import FindTechniciansExecutor
from slot_extractor.tool_loop.fixture_store import FixtureStore
from slot_extractor.tool_loop.models import ToolQuery

EXECUTOR = FindTechniciansExecutor(
    FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml"))
)


def query(**overrides):
    values = dict(
        technician_name=None,
        start_time=datetime(2026, 8, 13, 15),
        duration_minutes=60,
        gender_preference="female",
        preferences=("肩颈",),
    )
    return ToolQuery(**(values | overrides))


def test_specific_available_unavailable_and_not_found():
    assert EXECUTOR.find(query(technician_name="王芳")).status == "available"
    assert EXECUTOR.find(query(technician_name="李明")).status == "unavailable"
    result = EXECUTOR.find(query(technician_name="陈静"))
    assert result.status == "not_found" and result.candidates == ()


def test_search_match_no_match_and_unmodelled_preferences():
    result = EXECUTOR.find(query())
    assert result.status == "matched" and result.candidates[0].name == "王芳"
    assert EXECUTOR.find(query(preferences=("泰式",))).status == "no_match"
    assert EXECUTOR.find(query(preferences=("轻柔",))).status == "mock_coverage_miss"


def test_dataset_specialty_aliases_are_normalized():
    result = EXECUTOR.find(query(technician_name="王芳", preferences=("肩颈按摩",)))
    assert result.status == "available"


def test_strength_preference_alias_is_modelled():
    result = EXECUTOR.find(
        query(technician_name="王芳", gender_preference=None, preferences=("力气大",))
    )
    assert result.status == "available"


def test_generic_massage_does_not_filter_technician_specialties():
    result = EXECUTOR.find(query(technician_name="王芳", preferences=("按摩",)))
    assert result.status == "available"


def test_end_boundary_is_valid_but_one_minute_beyond_is_not():
    assert (
        EXECUTOR.find(query(technician_name="王芳", start_time=datetime(2026, 8, 13, 17))).status
        == "available"
    )
    assert (
        EXECUTOR.find(query(technician_name="王芳", start_time=datetime(2026, 8, 13, 17, 1))).status
        == "unavailable"
    )


def test_ambiguous_search_is_explicit_coverage_miss():
    result = EXECUTOR.find(
        query(start_time=datetime(2026, 8, 13, 10), gender_preference=None, preferences=("精油",))
    )
    assert result.status == "mock_coverage_miss"
    assert result.error_code == "ambiguous_candidates"
    assert len(result.trace) == 2
