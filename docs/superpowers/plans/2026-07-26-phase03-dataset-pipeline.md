# 阶段三：SFT + DPO 训练数据集构造管线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可离线复现的 raw → 校验 → SFT/DPO → 分层切分 → 训评隔离 → LLaMA-Factory 注册管线，并用每类 5 条、共 25 条 raw 完成阶段三小样冒烟。

**Architecture:** 以一份带 `dpo_targets` 的 raw 数据作为唯一事实源；生成器只负责生产 raw，校验器在入库前落实业务合同，SFT 渲染器和 DPO 扰动器只做确定性转换。`dataset_build` 聚合切分、审计、隔离和版本登记，CLI 仅负责参数解析与 I/O，使各模块均可单测并允许后续把 mock 后端替换为 GPT-5.6-sol。

**Tech Stack:** Python 3.12、标准库 `dataclasses/json/hashlib/random/argparse/pathlib`、现有 Backend/PromptBuilder/JSONL 工具、pytest、ruff、LLaMA-Factory ShareGPT/ShareGPT preference 数据格式。

---

## 范围与文件结构

本计划实现设计文档 `docs/superpowers/specs/2026-07-22-phase03-dataset-design.md` 的“小样冒烟”范围。约 1,400 条 SFT / 400 对 DPO 的全量生成、LLaMA-Factory 实际训练、LLM 版 DPO 扰动和失败样本回流均不在本计划内。

**新增文件：**

- `src/slot_extractor/data/raw_sample.py`：raw 数据类、类别/痛点白名单与解析入口。
- `src/slot_extractor/data/generator.py`：规格 prompt、Backend 调用、响应解析。
- `src/slot_extractor/data/raw_validator.py`：schema、跨字段一致性和来源事实校验。
- `src/slot_extractor/data/tool_schema.py`：把 raw 的可用工具名映射为稳定的 OpenAI-compatible JSON Schema。
- `src/slot_extractor/data/sft_render.py`：复用 PromptBuilder 生成 ShareGPT SFT 记录并保持消息角色。
- `src/slot_extractor/data/fake_names.py`：与工具结果排斥的固定假名池。
- `src/slot_extractor/data/dpo_perturb.py`：P4/P6/P5/P7/P2P3 确定性扰动。
- `src/slot_extractor/data/isolation.py`：规范化输入指纹及训评零重叠检查。
- `src/slot_extractor/data/tag_audit.py`：分类计数和类内难例比例审计。
- `src/slot_extractor/data/dataset_build.py`：分层切分、产物写入、版本卡和注册信息。
- `scripts/data/build_dataset.py`：阶段三命令行入口。
- `configs/data/phase03.yaml`：版本、配比、阈值和路径配置。
- `configs/inference/mock_phase03.yaml`：25 条确定性 mock raw 响应。
- `configs/training/llamafactory/VERSION`：锁定数据合同与训练环境版本 `v0.9.5`。
- `tests/fixtures/phase03_raw.jsonl`：五类合法 raw fixture。
- `tests/unit/test_raw_sample.py`、`test_raw_validator.py`、`test_generator.py`、`test_sft_render.py`、`test_dpo_perturb.py`、`test_isolation.py`、`test_tag_audit.py`、`test_dataset_build.py`。
- `tests/integration/test_pipeline_phase03.py`：25 条端到端冒烟。

**修改文件：**

- `src/slot_extractor/schemas/sample.py`：把 history/context 校验提升为公开复用函数。
- `src/slot_extractor/utils/jsonl.py`：增加统一 JSONL 写入函数。
- `src/slot_extractor/data/__init__.py`、`scripts/data/__init__.py`：导出公共入口。
- `pyproject.toml`：注册 `slot-build-dataset` 命令。
- `project-log/phase-03-dataset/log.md`：登记交付、命令、产物和未执行项。

运行测试时不得改写已冻结的 `data/eval/test.jsonl` 或 `data/eval/test.sha256`；集成测试全部写入 pytest 的 `tmp_path`。

### Task 1: 建立 raw 样本合同并公开上下文校验器

**Files:**
- Create: `src/slot_extractor/data/raw_sample.py`
- Modify: `src/slot_extractor/schemas/sample.py`
- Create: `tests/unit/test_raw_sample.py`

- [ ] **Step 1: 写失败测试，固定 raw 的七字段、类别与 DPO 白名单**

```python
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record

def test_parse_raw_sample(ask_raw_record):
    sample = raw_sample_from_record(ask_raw_record)
    assert isinstance(sample, RawSample)
    assert sample.category == "追问"
    assert sample.dpo_targets == ("P7",)

def test_reject_out_of_category_dpo_target(ask_raw_record):
    ask_raw_record["dpo_targets"] = ["P4"]
    with pytest.raises(ValueError, match="dpo_targets"):
        raw_sample_from_record(ask_raw_record)
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `uv run pytest tests/unit/test_raw_sample.py -v`
Expected: FAIL，包含 `ModuleNotFoundError: slot_extractor.data.raw_sample`。

- [ ] **Step 3: 将 `_validate_history/_validate_context` 重命名为公开函数**

在 `sample.py` 定义 `validate_history(sample_id, input_obj)` 和 `validate_context(sample_id, input_obj)`，并让 `sample_from_record` 调用新名称；不要改变既有验证语义。

- [ ] **Step 4: 实现 raw 数据类与解析器**

```python
CATEGORY_TAGS = {"追问", "工具调用", "最终 JSON", "确认", "无关"}
DPO_TARGETS_BY_CATEGORY = {
    "追问": frozenset({"P7"}),
    "工具调用": frozenset({"P6", "P2P3"}),
    "最终 JSON": frozenset({"P4", "P2P3"}),
    "确认": frozenset({"P5"}),
    "无关": frozenset({"P5", "P6"}),
}

@dataclass(frozen=True)
class RawSample:
    id: str
    output_kind: Literal["final", "tool_call"]
    conversation_kind: Literal["single_turn", "multi_turn"]
    tags: tuple[str, ...]
    input: dict[str, Any]
    expected: dict[str, Any]
    dpo_targets: tuple[str, ...]

    @property
    def category(self) -> str:
        return next(tag for tag in self.tags if tag in CATEGORY_TAGS)
```

`raw_sample_from_record` 必须要求顶层字段集合严格等于七字段，恰有一个类别 tag，`expected.action == output_kind`，`dpo_targets` 无重复且属于类别白名单，并调用公开的 history/context 校验。

- [ ] **Step 5: 运行 raw 与既有样本加载测试**

Run: `uv run pytest tests/unit/test_raw_sample.py tests/unit/test_sample_loader.py -v`
Expected: PASS；证明公开化没有破坏评估集加载。

- [ ] **Step 6: 提交合同层**

```powershell
git add src/slot_extractor/data/raw_sample.py src/slot_extractor/schemas/sample.py tests/unit/test_raw_sample.py
git commit -m "feat(data): define phase03 raw sample contract"
```

### Task 2: 实现 raw 三级校验的第一层

**Files:**
- Create: `src/slot_extractor/data/raw_validator.py`
- Create: `tests/unit/test_raw_validator.py`
- Create: `tests/fixtures/phase03_raw.jsonl`

- [ ] **Step 1: 写失败测试覆盖 schema 和跨字段红线**

```python
def test_validate_valid_raw(final_tool_result_raw):
    validate_raw_sample(final_tool_result_raw)

@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.expected.update(missing_info=[]), "missing_info"),
    (lambda r: r.expected.update(info_complete=False), "info_complete"),
    (lambda r: r.expected.update(technician_name="工具外姓名"), "tool result"),
])
def test_reject_inconsistent_final(final_tool_result_raw, mutate, message):
    mutate(final_tool_result_raw)
    with pytest.raises(RawValidationError, match=message):
        validate_raw_sample(final_tool_result_raw)
```

另加 unrelated、confirmation、`not_found/no_match` 姓名必须为空、对话轮数计算、非法时间和 tool_call schema 用例。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_raw_validator.py -v`
Expected: FAIL，缺少 `raw_validator`。

- [ ] **Step 3: 实现 schema 分派与跨字段验证**

```python
def validate_raw_sample(sample: RawSample) -> None:
    if sample.output_kind == "final":
        validate_final_output(sample.expected)
        _validate_final_consistency(sample)
    else:
        validate_tool_call_output(sample.expected)
    _validate_conversation_kind(sample)
```

`_validate_final_consistency` 精确落实设计文档 5.1：缺失字段由 `start_time/duration_minutes` 推导；`info_complete` 与二者齐全性相等；无关样本槽位为 null/默认且 handoff；确认必须完整；工具结果 final 的姓名必须出现在最近 tool JSON 中；状态、姓名和 reply_type 一致。

- [ ] **Step 4: 建立五类 fixture**

`tests/fixtures/phase03_raw.jsonl` 至少各含一条追问、工具调用、最终 JSON、确认、无关样本；字段直接采用阶段三设计文档五张规格卡的完整合法结构，不加入评估专用的 `assertions/reply_expectations`。

- [ ] **Step 5: 运行校验测试**

Run: `uv run pytest tests/unit/test_raw_validator.py -v`
Expected: PASS。

- [ ] **Step 6: 提交校验层**

```powershell
git add src/slot_extractor/data/raw_validator.py tests/unit/test_raw_validator.py tests/fixtures/phase03_raw.jsonl
git commit -m "feat(data): validate raw training samples"
```

### Task 3: 实现强模型生成器和离线 mock 接口

**Files:**
- Create: `src/slot_extractor/data/generator.py`
- Create: `tests/unit/test_generator.py`
- Create: `configs/inference/mock_phase03.yaml`

- [ ] **Step 1: 写失败测试固定一次请求、JSON 解析和失败语义**

```python
def test_generate_one_validates_backend_json(valid_record):
    backend = StubBackend(json.dumps(valid_record, ensure_ascii=False))
    sample = RawGenerator(backend).generate_one(GenerationRequest("追问", 1))
    assert sample.id == valid_record["id"]
    assert backend.calls[0][-1]["role"] == "user"

def test_generate_one_rejects_markdown_fence():
    backend = StubBackend("```json\n{}\n```")
    with pytest.raises(GenerationError, match="raw JSON"):
        RawGenerator(backend).generate_one(GenerationRequest("追问", 1))
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `uv run pytest tests/unit/test_generator.py -v`
Expected: FAIL，缺少 `generator`。

- [ ] **Step 3: 实现请求和生成循环**

```python
@dataclass(frozen=True)
class GenerationRequest:
    category: str
    count: int

class RawGenerator:
    def __init__(self, backend: Backend): self.backend = backend
    def generate_one(self, request: GenerationRequest) -> RawSample:
        result = self.backend.generate(build_generation_messages(request))
        record = parse_raw_json(result.text)
        sample = raw_sample_from_record(record)
        validate_raw_sample(sample)
        return sample
```

生成 prompt 必须包含：七字段合同、指定类别白名单、`current_state` 规则、五类目标数量、只输出单个 JSON 对象；`generate_many` 按类别和序号调用，并拒绝重复 id。

- [ ] **Step 4: 配置 25 条 mock 响应**

`configs/inference/mock_phase03.yaml` 使用现有 `MockBackend` 格式，key 为确定性请求 id（如 `phase03-ask-001`）；每类 5 条，覆盖 single/multi-turn、tool result、确认和无关边界。响应 `text` 是完整 raw JSON 字符串。

- [ ] **Step 5: 运行生成器测试**

Run: `uv run pytest tests/unit/test_generator.py -v`
Expected: PASS。

- [ ] **Step 6: 提交生成层**

```powershell
git add src/slot_extractor/data/generator.py tests/unit/test_generator.py configs/inference/mock_phase03.yaml
git commit -m "feat(data): add raw sample generator"
```

### Task 4: 实现与部署 messages 同源的 ShareGPT SFT 渲染器

**Files:**
- Create: `src/slot_extractor/data/tool_schema.py`
- Create: `src/slot_extractor/data/sft_render.py`
- Modify: `src/slot_extractor/prompts/template.py`
- Create: `tests/unit/test_sft_render.py`
- Modify: `tests/unit/test_prompt_builder.py`

- [ ] **Step 1: 写失败测试固定 ShareGPT 格式、角色边界和 tool 事件保留**

```python
def test_render_sft_keeps_system_and_target_role(ask_raw):
    row = render_sft(ask_raw)
    assert set(row) == {"system", "tools", "conversations"}
    assert "当前状态：null" in row["system"]
    assert json.loads(row["tools"])[0]["name"] == "find_technicians"
    assert row["conversations"][-1]["from"] == "gpt"
    assert json.loads(row["conversations"][-1]["value"]) == ask_raw.expected

def test_render_sft_preserves_tool_event(final_tool_result_raw):
    row = render_sft(final_tool_result_raw)
    roles = [message["from"] for message in row["conversations"]]
    assert roles[-3:] == ["function_call", "observation", "gpt"]

def test_tool_call_target_remains_gpt_json(tool_call_raw):
    row = render_sft(tool_call_raw)
    assert row["conversations"][-1]["from"] == "gpt"
    assert json.loads(row["conversations"][-1]["value"])["action"] == "tool_call"

def test_sharegpt_roles_follow_llamafactory_positions(final_tool_result_raw):
    row = render_sft(final_tool_result_raw)
    odd = {"human", "observation"}
    even = {"gpt", "function_call"}
    assert all(message["from"] in (odd if index % 2 == 0 else even)
               for index, message in enumerate(row["conversations"]))
```

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_sft_render.py -v`
Expected: FAIL，缺少 `sft_render`。

- [ ] **Step 3: 实现稳定工具 Schema 注册表**

`tool_schema.py` 定义 `find_technicians` 的 OpenAI-compatible schema（name、description、parameters.type/object、五个 properties 和 required），并提供 `render_tools(available_tools) -> str`。按 raw 列表顺序选择已知工具，未知工具抛 `ValueError`，使用紧凑 JSON 字符串输出；空列表输出 `"[]"`。schema 内容必须与 `prompts/rules.py` 的字段合同一致。

- [ ] **Step 4: 实现 raw 到 PromptBuilder Sample 的适配和角色映射**

```python
def render_sft(raw: RawSample) -> dict[str, object]:
    sample = Sample(raw.id, raw.output_kind, raw.conversation_kind,
                    raw.input, raw.expected, [], list(raw.tags))
    system, *turns = PromptBuilder().build_messages(
        sample, include_tool_descriptions=False
    )
    return {
        "system": str(system["content"]),
        "tools": render_tools(raw.input.get("available_tools", [])),
        "conversations": [
            *messages_to_sharegpt(turns),
            {"from": "gpt", "value": compact_json(raw.expected)},
        ],
    }
```

给 `PromptBuilder.build_messages` 增加仅限关键字参数 `include_tool_descriptions: bool = True`；默认值保证现有推理路径不变，ShareGPT 渲染时传 `False`，使 system 只保留规则、output schema、当前时间和当前状态，工具定义只进入顶层 `tools` 而不重复。`messages_to_sharegpt` 按 `user→human`、自然语言 `assistant→gpt`、`assistant.tool_calls→function_call`、`tool→observation` 映射；function_call 的 value 固定为 `{"name":<工具名>,"arguments":<参数对象>}` 紧凑 JSON，observation 的 value 保留工具结果 JSON。不得把 `current_state` 变成独立角色或在 conversations 中重复；最后追加的目标 `gpt` 不属于 history。映射完成后验证 `human/observation` 位于零基偶数索引（LLaMA-Factory 文档的奇数位），`gpt/function_call` 位于零基奇数索引，否则抛 `ShareGPTFormatError`。在 `test_prompt_builder.py` 增加默认仍含工具描述、显式 False 时不含工具描述的两条回归断言。

当前目标无论 `expected.action` 是 `final` 还是 `tool_call` 都必须追加为 `from=gpt` 的紧凑业务 JSON；只有 raw history 中已经发生的 `assistant.tool_calls` 才映射为 `function_call`。不得改变现有模型输出合同。

- [ ] **Step 5: 运行 prompt 与 SFT 回归测试**

Run: `uv run pytest tests/unit/test_sft_render.py tests/unit/test_prompt_builder.py -v`
Expected: PASS。

- [ ] **Step 6: 提交 SFT 渲染器**

```powershell
git add src/slot_extractor/data/tool_schema.py src/slot_extractor/data/sft_render.py src/slot_extractor/prompts/template.py tests/unit/test_sft_render.py tests/unit/test_prompt_builder.py
git commit -m "feat(data): render raw samples as sharegpt sft"
```

阶段三不在此任务中修改在线 Backend；但 `tool_schema.py` 是训练与后续推理共享的唯一工具 schema 源。阶段四空跑计划必须先让本地推理 Backend 接收并传递该 schema，禁止另写一份工具定义。

### Task 5: 实现五类确定性 DPO 扰动

**Files:**
- Create: `src/slot_extractor/data/fake_names.py`
- Create: `src/slot_extractor/data/dpo_perturb.py`
- Create: `tests/unit/test_dpo_perturb.py`

- [ ] **Step 1: 写参数化失败测试覆盖五个痛点**

```python
@pytest.mark.parametrize("target", ["P4", "P6", "P5", "P7", "P2P3"])
def test_perturbation_is_valid_and_different(raw_for_target, target):
    pair = perturb(raw_for_target[target], target)
    chosen, rejected = map(json.loads, (pair["chosen"], pair["rejected"]))
    validate_output(chosen)
    validate_output(rejected)
    assert chosen != rejected
```

增加断言：P4 假名不在 tool 结果；P6 整体切换 schema；P5 确认翻转或回退追问；P7 补齐 14:00/60；P2P3 对 null 枚举回退到时间偏移。

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_dpo_perturb.py -v`
Expected: FAIL，缺少 `dpo_perturb`。

- [ ] **Step 3: 建立统一输出校验和 pair 包装**

```python
PERTURBERS = {"P4": perturb_p4, "P6": perturb_p6,
              "P5": perturb_p5, "P7": perturb_p7,
              "P2P3": perturb_p2p3}

def perturb(raw: RawSample, target: str) -> dict[str, object]:
    if target not in raw.dpo_targets:
        raise PerturbationError(f"{raw.id} does not declare {target}")
    rejected = PERTURBERS[target](raw)
    validate_output(rejected)
    if rejected == raw.expected:
        raise PerturbationError("rejected equals chosen")
    base = render_sft(raw)
    context = base["conversations"][:-1]
    return {"system": base["system"], "tools": base["tools"],
            "conversations": context,
            "chosen": {"from": "gpt", "value": compact_json(raw.expected)},
            "rejected": {"from": "gpt", "value": compact_json(rejected)}}
```

- [ ] **Step 4: 逐个实现设计文档 3.2.2 的算法**

使用固定 `FAKE_NAMES` 和按 sample id 派生的稳定索引，不用全局随机状态。final/tool_call 互换时从零创建目标 schema；时间使用 `datetime.strptime` 后 `timedelta(days=1)`；duration 先 `+30`；枚举仅在非 null 时翻转。每个函数只实现一个痛点，不共享可变字典，先 `deepcopy` chosen。

- [ ] **Step 5: 运行扰动测试**

Run: `uv run pytest tests/unit/test_dpo_perturb.py -v`
Expected: PASS，五类 chosen/rejected 均过现有 output schema 且不相等。

- [ ] **Step 6: 提交 DPO 层**

```powershell
git add src/slot_extractor/data/fake_names.py src/slot_extractor/data/dpo_perturb.py tests/unit/test_dpo_perturb.py
git commit -m "feat(data): derive deterministic dpo preference pairs"
```

### Task 6: 实现训评指纹隔离和难例审计

**Files:**
- Create: `src/slot_extractor/data/isolation.py`
- Create: `src/slot_extractor/data/tag_audit.py`
- Create: `tests/unit/test_isolation.py`
- Create: `tests/unit/test_tag_audit.py`

- [ ] **Step 1: 写失败测试固定规范化与报警输出**

```python
def test_fingerprint_ignores_json_key_order(record):
    reordered = deepcopy(record)
    reordered["input"] = dict(reversed(list(reordered["input"].items())))
    assert input_fingerprint(record) == input_fingerprint(reordered)

def test_overlap_is_rejected(train_record, eval_record):
    eval_record["input"] = deepcopy(train_record["input"])
    with pytest.raises(IsolationError, match=train_record["id"]):
        assert_no_eval_overlap([train_record], [eval_record])

def test_audit_reports_category_deficit(samples):
    report = audit_tags(samples, {"工具调用": 0.60})
    assert report.deficits["工具调用"].required_ratio == 0.60
```

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_isolation.py tests/unit/test_tag_audit.py -v`
Expected: FAIL，两个模块不存在。

- [ ] **Step 3: 实现统一输入指纹**

指纹 payload 只能由 `history/user_input/current_time` 构成；递归规范化字符串（strip、合并连续空白），对象 `sort_keys=True`，数组保持顺序，UTF-8 后取 SHA-256。错误必须列出训练 id、评估 id 和 hash。

- [ ] **Step 4: 实现 tag 审计报告**

难例 tag 集合固定为 `相对时间/多义短词/多轮改口/易混边界/幻觉陷阱`；按主类别汇总 total、hard、ratio，并对配置阈值生成结构化 deficits。小样不足只报警，不静默通过；build 的严格模式据此返回非零。

- [ ] **Step 5: 运行隔离与审计测试**

Run: `uv run pytest tests/unit/test_isolation.py tests/unit/test_tag_audit.py -v`
Expected: PASS。

- [ ] **Step 6: 提交质量门禁**

```powershell
git add src/slot_extractor/data/isolation.py src/slot_extractor/data/tag_audit.py tests/unit/test_isolation.py tests/unit/test_tag_audit.py
git commit -m "feat(data): enforce isolation and hard-case ratios"
```

### Task 7: 实现分层切分、版本产物和 LLaMA-Factory 注册

**Files:**
- Modify: `src/slot_extractor/utils/jsonl.py`
- Create: `src/slot_extractor/data/dataset_build.py`
- Create: `tests/unit/test_dataset_build.py`

- [ ] **Step 1: 写失败测试固定可复现切分和产物清单**

```python
def test_build_writes_expected_artifacts(raw_samples, tmp_path):
    result = build_dataset(raw_samples, eval_records=[], output_root=tmp_path,
                           version="v0.1", seed=42)
    assert result.sft_train.exists()
    assert result.sft_val.exists()
    assert result.dpo_pairs.exists()
    assert result.dataset_info.exists()
    assert result.version_card.exists()
```

另断言每类 5 条按固定 seed 切为 4 train + 1 val、重复运行文件字节一致、DPO 行数等于所有 `dpo_targets` 数量、SFT/DPO 注册均为 ShareGPT 且 role/content/tool tag 一致。

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_dataset_build.py -v`
Expected: FAIL，缺少 `dataset_build`。

- [ ] **Step 3: 给 JSONL 工具增加原子写入函数**

```python
def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(target)
```

- [ ] **Step 4: 实现 build 顺序和分层切分**

固定顺序为：验证全部 raw → 检查 eval 重叠 → 审计 tags → 按类别 `Random(seed)` 排序并 9:1 切分（每个非空类别至少 1 条 val）→ 渲染 SFT → 逐 target 派生 DPO → 写文件。任何门禁失败前不得留下半成品目录。

- [ ] **Step 5: 写注册文件和版本卡**

`dataset_info.json` 严格按设计文档 3.3 的完整 JSON 注册 `phase03_sft_v0_1` 与 `phase03_dpo_v0_1`：两者均设置 `formatting: sharegpt`，columns 映射 `messages: conversations`、`system: system`、`tools: tools`；DPO 额外映射 `chosen/rejected` 并设置 `ranking: true`。两者共用 tags：`role_tag: from`、`content_tag: value`、`user_tag: human`、`assistant_tag: gpt`、`function_tag: function_call`、`observation_tag: observation`。测试必须逐字段等值比较该合同，不只检查文件存在。版本卡记录版本、UTC 构建时间、git commit（脏工作区附 `-dirty`）、模型/配置、raw/SFT/DPO 数量、分类和痛点分布、seed、隔离结果与审计结果。

- [ ] **Step 6: 运行构建器测试**

Run: `uv run pytest tests/unit/test_dataset_build.py tests/unit/test_isolation.py tests/unit/test_tag_audit.py -v`
Expected: PASS。

- [ ] **Step 7: 提交构建层**

```powershell
git add src/slot_extractor/utils/jsonl.py src/slot_extractor/data/dataset_build.py tests/unit/test_dataset_build.py
git commit -m "feat(data): build versioned sft and dpo datasets"
```

### Task 8: 实现 CLI 与 25 条离线端到端冒烟

**Files:**
- Create: `configs/data/phase03.yaml`
- Create: `configs/training/llamafactory/VERSION`
- Create: `scripts/data/build_dataset.py`
- Modify: `scripts/data/__init__.py`
- Modify: `src/slot_extractor/data/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/integration/test_pipeline_phase03.py`

- [ ] **Step 1: 写失败集成测试**

```python
def test_mock_pipeline_builds_25_samples(tmp_path):
    completed = subprocess.run([
        sys.executable, "-m", "scripts.data.build_dataset", "--mock",
        "--config", "configs/data/phase03.yaml",
        "--output-root", str(tmp_path),
    ], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert len(list(read_jsonl(tmp_path / "raw/v0.1/samples.jsonl"))) == 25
    assert len(list(read_jsonl(tmp_path / "processed/sft/v0.1/train.jsonl"))) == 20
    assert len(list(read_jsonl(tmp_path / "processed/sft/v0.1/val.jsonl"))) == 5
```

- [ ] **Step 2: 运行并确认 CLI 缺失**

Run: `uv run pytest tests/integration/test_pipeline_phase03.py -v`
Expected: FAIL，`scripts.data.build_dataset` 不存在。

- [ ] **Step 3: 写阶段配置**

`phase03.yaml` 明确写出 `version: v0.1`、五类各 5 条、`seed: 42`、五类难例阈值、eval 路径、mock/真实 inference 配置路径、`formatting: sharegpt` 以及默认 output root；不在配置中放 API key。阶段四训练 YAML 必须据此设置 `train_on_prompt: false`、`mask_history: true` 和目标模型对应的 `template`。

同时创建 `configs/training/llamafactory/VERSION`，内容严格为 `v0.9.5` 加换行；阶段日志记录官方 release URL。环境安装命令固定为 `uv tool install "llamafactory==0.9.5"`，不得使用无版本约束的安装命令。

- [ ] **Step 4: 实现 CLI**

支持 `--config`、`--output-root`、互斥的 `--mock/--generate`、`--raw-input`、`--strict-audit`。`--mock` 使用 mock 配置生成 25 条；`--generate` 使用 GPT-5.6-sol 配置；`--raw-input` 跳过生成、重建确定性产物。最终 stdout 打印 raw/train/val/DPO 数量和所有产物路径，异常写 stderr 并返回 1。

- [ ] **Step 5: 注册命令并导出入口**

在 `pyproject.toml` 的 `[project.scripts]` 增加：

```toml
slot-build-dataset = "scripts.data.build_dataset:main"
```

- [ ] **Step 6: 跑离线端到端测试**

Run: `uv run pytest tests/integration/test_pipeline_phase03.py -v`
Expected: PASS；raw=25、SFT train=20、val=5，DPO 数量与 fixture 声明 targets 总数一致，dataset_info 和版本卡存在。

- [ ] **Step 7: 提交 CLI 和冒烟**

```powershell
git add configs/data/phase03.yaml configs/training/llamafactory/VERSION scripts/data/build_dataset.py scripts/data/__init__.py src/slot_extractor/data/__init__.py pyproject.toml tests/integration/test_pipeline_phase03.py
git commit -m "feat(data): add phase03 dataset build cli"
```

### Task 9: 生成小样产物、验证并完成阶段日志

**Files:**
- Create: `data/raw/v0.1/samples.jsonl`
- Create: `data/processed/sft/v0.1/train.jsonl`
- Create: `data/processed/sft/v0.1/val.jsonl`
- Create: `data/processed/dpo/v0.1/train.jsonl`
- Create: `data/processed/dpo/v0.1/val.jsonl`
- Create: `data/processed/v0.1/dataset_info.json`
- Create: `data/processed/v0.1/DATASET_CARD.md`
- Modify: `project-log/phase-03-dataset/log.md`

- [ ] **Step 1: 执行 mock 小样构建**

Run: `uv run slot-build-dataset --mock --config configs/data/phase03.yaml`
Expected: exit 0，打印 `raw=25, sft_train=20, sft_val=5` 及实际 DPO 对数；隔离检查为 0 overlap。

- [ ] **Step 2: 重跑并验证可复现**

先记录所有产物 SHA-256，再用相同命令重跑并再次计算。

Run: `Get-FileHash data/raw/v0.1/samples.jsonl,data/processed/sft/v0.1/train.jsonl,data/processed/sft/v0.1/val.jsonl,data/processed/dpo/v0.1/train.jsonl,data/processed/dpo/v0.1/val.jsonl -Algorithm SHA256`
Expected: 两次 hash 完全相同。

- [ ] **Step 3: 运行阶段三聚焦测试**

Run: `uv run pytest tests/unit/test_raw_sample.py tests/unit/test_raw_validator.py tests/unit/test_generator.py tests/unit/test_sft_render.py tests/unit/test_dpo_perturb.py tests/unit/test_isolation.py tests/unit/test_tag_audit.py tests/unit/test_dataset_build.py tests/integration/test_pipeline_phase03.py -v`
Expected: PASS。

- [ ] **Step 4: 运行全量非本地测试和静态检查**

Run: `uv run pytest -m "not local_backend"`
Expected: PASS，无失败或错误。

Run: `uv run ruff check .`
Expected: `All checks passed!`。

- [ ] **Step 5: 真实 GPT-5.6-sol 生成 25 条（仅凭据已配置时）**

Run: `uv run slot-build-dataset --generate --config configs/data/phase03.yaml --output-root experiments/runs/phase03-gpt-smoke`
Expected: exit 0，生成 25 条且全部通过同一校验、隔离和渲染管线。若环境没有 `OPENAI_BASE_URL/OPENAI_API_KEY`，不得伪称已验证；在日志明确记录“未执行：缺少运行凭据”，mock DoD 仍可独立完成。

- [ ] **Step 6: 更新阶段日志和数据卡**

把 `project-log/phase-03-dataset/log.md` 的现有占位内容和开放问题替换为实际日期、commit、命令、计数、审计比例、隔离结果、mock/真机状态、产物路径；`DATASET_CARD.md` 说明小样非正式训练集、数据来源、字段、切分、DPO 路由、限制和后续全量生成命令。

- [ ] **Step 7: 提交阶段三交付物**

```powershell
git add data/raw/v0.1 data/processed project-log/phase-03-dataset/log.md
git commit -m "docs(data): record phase03 smoke dataset build"
```

## 自检结论

- 规格覆盖：生成器、raw 合同、三级校验、五类 SFT、五类 DPO、难例审计、9:1 分层切分、训评隔离、版本卡、LLaMA-Factory 注册、mock 25 条和可选真机 25 条均有对应任务。
- 范围控制：没有纳入全量约 1,400/400 数据生成、训练、LLM 扰动或数据闭环。
- 类型一致：全计划统一使用 `RawSample`、`dpo_targets: tuple[str, ...]`、五个痛点 token；SFT 使用 `system/tools/conversations`，DPO 使用相同共同上下文和消息对象形式的 `chosen/rejected`；工具调用角色统一为官方 `function_call`。
- 占位扫描：实现步骤没有待补内容；真实 API 凭据不可用时的客观分支已明确为日志记录，而不是伪造成功。
