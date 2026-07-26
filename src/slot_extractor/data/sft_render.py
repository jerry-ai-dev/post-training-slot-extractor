from __future__ import annotations

import json
from typing import Any

from slot_extractor.data.raw_sample import RawSample
from slot_extractor.data.tool_schema import render_tools
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import Sample


class ShareGPTFormatError(ValueError):
    pass


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def messages_to_sharegpt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            result.append({"from": "human", "value": str(message["content"])})
        elif role == "assistant" and "tool_calls" in message:
            call = message["tool_calls"][0]["function"]
            result.append(
                {
                    "from": "function_call",
                    "value": compact_json(
                        {"name": call["name"], "arguments": json.loads(call["arguments"])}
                    ),
                }
            )
        elif role == "assistant":
            result.append({"from": "gpt", "value": str(message["content"])})
        elif role == "tool":
            result.append({"from": "observation", "value": str(message["content"])})
    return result


def render_sft(raw: RawSample) -> dict[str, object]:
    sample = Sample(
        raw.id, raw.output_kind, raw.conversation_kind, raw.input, raw.expected, [], list(raw.tags)
    )
    system, *turns = PromptBuilder().build_messages(sample, include_tool_descriptions=False)
    conversations = messages_to_sharegpt(turns)
    conversations.append({"from": "gpt", "value": compact_json(raw.expected)})
    odd, even = {"human", "observation"}, {"gpt", "function_call"}
    if any(
        message["from"] not in (odd if index % 2 == 0 else even)
        for index, message in enumerate(conversations)
    ):
        raise ShareGPTFormatError("ShareGPT role positions are invalid")
    tools = raw.input.get("available_tools", [])
    if not isinstance(tools, list) or not all(isinstance(item, str) for item in tools):
        raise ValueError("available_tools must be a string list")
    return {
        "system": str(system["content"]),
        "tools": render_tools(tools),
        "conversations": conversations,
    }
