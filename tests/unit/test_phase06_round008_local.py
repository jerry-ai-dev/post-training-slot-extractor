from pathlib import Path

from scripts.eval.run_phase06_round008_local import CompactPromptBuilder
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import load_samples


def test_compact_prompt_is_materially_shorter() -> None:
    sample = load_samples(Path("data/eval/phase06_holdout_v0.3.jsonl"))[0]
    full = PromptBuilder().build_messages(sample)[0]["content"]
    compact = CompactPromptBuilder().build_messages(sample)[0]["content"]
    assert isinstance(full, str)
    assert isinstance(compact, str)
    assert len(compact) < len(full) * 0.7


def test_compact_prompt_keeps_protocol_contracts() -> None:
    sample = load_samples(Path("data/eval/phase06_holdout_v0.3.jsonl"))[0]
    system = CompactPromptBuilder().build_messages(sample)[0]["content"]
    assert isinstance(system, str)
    assert "tool_call" in system
    assert "missing_info" in system
    assert "当前时间" in system
    assert "当前状态" in system
