from dataclasses import replace

from slot_extractor.quantization.lineage import Lineage, cache_key


def test_cache_key_is_deterministic_and_changes_with_inputs():
    lineage = Lineage(
        model_id="model",
        base_model="base",
        base_revision="main",
        parent_model_id=None,
        adapter_run_id=None,
        source_sha256=(("base", "abc"),),
        git_revision="deadbeef",
        tool_versions=(("llama-quantize", "1"),),
    )
    first = cache_key(lineage, "quantize", {"type": "Q4_K_M", "threads": "8"})

    assert first == cache_key(
        lineage, "quantize", {"threads": "8", "type": "Q4_K_M"}
    )
    assert first != cache_key(
        replace(lineage, source_sha256=(("base", "changed"),)),
        "quantize",
        {"type": "Q4_K_M", "threads": "8"},
    )
    assert first != cache_key(
        replace(lineage, tool_versions=(("llama-quantize", "new"),)),
        "quantize",
        {"type": "Q4_K_M", "threads": "8"},
    )
