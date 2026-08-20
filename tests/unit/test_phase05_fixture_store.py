from datetime import date, datetime
from pathlib import Path

from slot_extractor.tool_loop.fixture_store import FixtureStore


def test_fixture_is_versioned_validated_and_contains_named_technicians():
    store = FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml"))
    assert store.version == "phase05-v1"
    assert {tech.name for tech in store.technicians()} == {"王芳", "李明"}
    wang = next(tech for tech in store.technicians() if tech.name == "王芳")
    assert wang.availability[0].contains(
        datetime(2026, 8, 13, 9), datetime(2026, 8, 13, 12)
    )


def test_fixture_can_shift_first_availability_day_without_changing_hash():
    path = Path("data/fixtures/technicians/phase05-v1.yaml")
    original = FixtureStore.from_yaml(path)
    shifted = FixtureStore.from_yaml(path, target_date=date(2026, 8, 20))

    assert shifted.date == "2026-08-20"
    assert shifted.fixture_hash == original.fixture_hash
    wang = next(tech for tech in shifted.technicians() if tech.name == "王芳")
    assert wang.availability[0].contains(
        datetime(2026, 8, 20, 9), datetime(2026, 8, 20, 12)
    )
    assert wang.availability[1].contains(
        datetime(2026, 8, 20, 14), datetime(2026, 8, 20, 18)
    )
    assert wang.availability[2].contains(
        datetime(2026, 8, 21, 10), datetime(2026, 8, 21, 17)
    )
