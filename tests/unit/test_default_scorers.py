# tests/unit/test_default_scorers.py
from slot_extractor.evaluation.runner import default_scorers


def test_default_scorers_registers_only_top_level_dimensions() -> None:
    scorers = {s.dimension: type(s).__name__ for s in default_scorers()}
    assert scorers == {
        "protocol": "InstructionScorer",
        "task_correctness": "TaskCorrectnessScorer",
        "resource": "NotAvailableScorer",
    }
