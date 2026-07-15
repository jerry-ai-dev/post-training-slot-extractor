# src/slot_extractor/prompts/template.py
from __future__ import annotations

import json

from slot_extractor.prompts.rules import (
    FINAL_SCHEMA_HINT,
    SYSTEM_RULES,
    TOOL_SCHEMA_HINT,
    render_tool_descriptions,
)
from slot_extractor.schemas.sample import Sample

Message = dict[str, object]


class PromptBuilder:
    """组装完整事件历史和结构化状态快照。

    结构：
      [system(规则+schema+工具+时间+状态)] + history + 可选 user
    """

    def build_messages(self, sample: Sample) -> list[Message]:
        available_tools = sample.input.get("available_tools")
        tool_descriptions = render_tool_descriptions(
            available_tools if isinstance(available_tools, list) else None
        )
        tool_block = (
            f"{TOOL_SCHEMA_HINT}\n{tool_descriptions}\n" if tool_descriptions else ""
        )
        current_time = sample.input.get("current_time", "")
        current_state = self._compact_json(sample.input.get("current_state"))
        system_content = (
            f"{SYSTEM_RULES}\n"
            f"{FINAL_SCHEMA_HINT}\n"
            f"{tool_block}"
            f"当前时间：{current_time}\n"
            f"当前状态：{current_state}"
        )

        messages: list[Message] = [
            {"role": "system", "content": system_content, "_sample_id": sample.id}
        ]

        messages.extend(self._history_turns(sample))

        user_input = sample.input.get("user_input")
        if isinstance(user_input, str) and user_input:
            messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _history_turns(sample: Sample) -> list[Message]:
        history = sample.input.get("history", [])
        if not isinstance(history, list):
            return []
        return [dict(turn) for turn in history if isinstance(turn, dict)]

    @staticmethod
    def _compact_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def messages_to_text(messages: list[Message]) -> str:
    """把 messages 数组扁平成可读文本（用于 mock 匹配 / 调试打印）。"""
    return "\n".join(
        f"[{message['role']}] "
        f"{message['tool_calls'] if 'tool_calls' in message else message.get('content')}"
        for message in messages
    )
