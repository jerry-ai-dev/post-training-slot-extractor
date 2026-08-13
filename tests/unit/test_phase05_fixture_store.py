from datetime import datetime
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
