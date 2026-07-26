from slot_extractor.data.raw_schema import raw_response_schema


def test_raw_schema_closes_top_level_and_requires_seven_fields() -> None:
    schema = raw_response_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "id",
        "output_kind",
        "conversation_kind",
        "tags",
        "input",
        "expected",
        "dpo_targets",
    }


def test_raw_schema_contains_exact_output_enums() -> None:
    schema = raw_response_schema()
    final, tool_call = schema["$defs"]["expected"]["anyOf"]
    assert final["properties"]["technician_status"]["enum"] == [
        "not_checked",
        "available",
        "unavailable",
        "not_found",
        "no_match",
    ]
    assert final["additionalProperties"] is False
    assert tool_call["additionalProperties"] is False
    assert set(final["required"]) == set(final["properties"])


def test_raw_schema_closes_history_shapes_and_dpo_tokens() -> None:
    schema = raw_response_schema()
    variants = schema["$defs"]["history_item"]["anyOf"]
    assert len(variants) == 4
    assert all(item["additionalProperties"] is False for item in variants)
    assert schema["properties"]["dpo_targets"]["items"]["enum"] == [
        "P4",
        "P6",
        "P5",
        "P7",
        "P2P3",
    ]


def test_every_enum_and_const_declares_type_for_responses_api() -> None:
    def visit(value):
        if isinstance(value, dict):
            if "enum" in value or "const" in value:
                assert "type" in value, value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw_response_schema())
