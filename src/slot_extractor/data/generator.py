from __future__ import annotations

import json
from dataclasses import dataclass

from slot_extractor.data.raw_sample import CATEGORY_TAGS, RawSample, raw_sample_from_record
from slot_extractor.data.raw_schema import raw_response_schema
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.data.scenario_specs import SCENARIOS, scenario_dpo_targets
from slot_extractor.inference.base import Backend, GenerationParams


class GenerationError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationRequest:
    category: str
    count: int
    scenario: str | None = None


def generation_sample_id(request: GenerationRequest) -> str:
    slug = {
        "追问": "ask",
        "工具调用": "tool",
        "最终 JSON": "final",
        "确认": "confirm",
        "无关": "unrelated",
    }[request.category]
    return f"phase03-{slug}-{request.count:03d}"


def build_generation_messages(request: GenerationRequest) -> list[dict[str, object]]:
    if request.category not in CATEGORY_TAGS or request.count < 1:
        raise GenerationError("invalid generation request")
    if request.scenario is not None:
        spec = SCENARIOS.get(request.scenario)
        if spec is None or spec.category != request.category:
            raise GenerationError("scenario does not belong to requested category")
    allowed = {
        "追问": "P7",
        "工具调用": "P6, P2P3",
        "最终 JSON": "P4, P2P3",
        "确认": "P5",
        "无关": "P5, P6",
    }[request.category]
    sample_id = generation_sample_id(request)
    hard_tag = {
        "追问": "相对时间",
        "工具调用": "易混边界",
        "最终 JSON": "幻觉陷阱",
        "确认": "多义短词",
        "无关": "易混边界",
    }[request.category]
    system = """你是预约信息抽取训练数据生成器。请生成一条高质量中文 raw 样本。
只输出单个原始 JSON 对象，禁止 Markdown、解释和代码围栏。
对象必须严格包含七个顶层字段且不得增删：id、output_kind、conversation_kind、tags、input、expected、dpo_targets。
output_kind 只能是 final/tool_call，且必须等于 expected.action。
conversation_kind 只能是 single_turn/multi_turn；tags 必须恰含一个主类别。
input 必须包含 history、current_time、current_state、available_tools，
并按场景包含 user_input；current_state 为对象或 null。
history 保留 user/assistant/tool 事件；工具调用 arguments 与工具结果 content 必须是 JSON 字符串。
history 中自然消息只能有 role/content 两个键；
assistant 工具消息只能有 role/content/tool_calls 三个键且 content=null；
工具结果只能有 role/tool_call_id/content 三个键。
final expected 必须且只能包含以下 14 字段：
action、gender_preference、technician_gender、start_time、duration_minutes、
preferences、technician_name、technician_status、confirmation、info_complete、
unrelated、missing_info、reply_type、reply。
tool_call expected 必须且只能包含 action、tool_name、arguments；
arguments 必须且只能包含 technician_name、start_time、duration_minutes、
gender_preference、preferences。
missing_info 仅由 start_time 和 duration_minutes 的缺失决定；info_complete 当且仅当两者齐全。
unrelated=true 时预约槽位为空或默认值、reply_type=handoff、reply=null；
confirmation=true 时信息必须完整。
技师姓名只能来自最近工具结果；not_found/no_match 时姓名必须为 null。
dpo_targets 必须从当前类别白名单选择，可为空，不得重复。生成自然、多样、可验证的边界样本。"""
    user = (
        f"生成类别：{request.category}\n"
        f"该类别 dpo_targets 白名单：{allowed}\n"
        f"请求序号：{request.count}\n"
        f"Sample ID: {sample_id}\n"
        f"id 必须严格使用 {sample_id}。\n"
        f'tags 必须严格为 ["{request.category}","{hard_tag}"]。\n'
        + {
            "追问": (
                "追问类按场景使用 single_turn 或 multi_turn；expected 为"
                " info_complete=false 的 final 追问，缺什么就准确填写 missing_info。"
            ),
            "工具调用": (
                "工具调用类按场景使用 single_turn 或 multi_turn；expected 为"
                " find_technicians tool_call，时间与时长必须已完整且使用绝对时间。"
            ),
            "最终 JSON": (
                "最终 JSON 类必须 multi_turn、不得有 user_input；history 严格依次为"
                " user 自然消息、assistant 单个 tool_calls、对应 tool 结果；expected 技师"
                "姓名必须来自该工具结果。"
            ),
            "确认": (
                "确认类按场景构造接受、拒绝或知悉结果；expected 为完整 final，"
                "confirmation 和 reply_type 必须严格服从场景说明。"
            ),
            "无关": (
                "无关类必须 single_turn、history=[]；expected 为 unrelated=true 的 final，"
                "预约槽位为空、missing_info=[]、reply_type=handoff、reply=null。"
            ),
        }[request.category]
        + (
            f"\n场景 ID：{request.scenario}\n"
            f"场景硬约束：{SCENARIOS[request.scenario].instruction}\n"
            f"dpo_targets 必须严格为 {list(scenario_dpo_targets(request.scenario))}。"
            if request.scenario
            else ""
        )
        + "\n只输出符合合同的 JSON。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_raw_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        raise GenerationError("backend must return raw JSON without Markdown fences")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise GenerationError("backend must return raw JSON") from exc
    if not isinstance(value, dict):
        raise GenerationError("backend raw JSON must be an object")
    return value


class RawGenerator:
    def __init__(self, backend: Backend, max_attempts: int = 3):
        self.backend = backend
        self.max_attempts = max_attempts

    def generate_one(self, request: GenerationRequest) -> RawSample:
        messages = build_generation_messages(request)
        params = GenerationParams(
            max_tokens=4096,
            response_schema=raw_response_schema(),
            response_schema_name="phase03_raw",
        )
        last_error: ValueError | None = None
        for _ in range(self.max_attempts):
            result = self.backend.generate(messages, params)
            try:
                record = parse_raw_json(result.text)
                if request.scenario:
                    record["dpo_targets"] = list(scenario_dpo_targets(request.scenario))
                sample = raw_sample_from_record(record)
                validate_raw_sample(sample)
                return sample
            except ValueError as exc:
                last_error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.text},
                    {
                        "role": "user",
                        "content": (
                            f"上一个 JSON 未通过校验：{exc}。请修正后重新输出完整的"
                            "单个原始 JSON 对象；禁止解释和 Markdown。"
                        ),
                    },
                ]
        sample_id = generation_sample_id(request)
        raise GenerationError(
            f"{sample_id} failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def generate_many(self, requests: list[GenerationRequest]) -> list[RawSample]:
        samples = [self.generate_one(request) for request in requests]
        ids = [sample.id for sample in samples]
        if len(ids) != len(set(ids)):
            raise GenerationError("duplicate raw sample id")
        return samples
