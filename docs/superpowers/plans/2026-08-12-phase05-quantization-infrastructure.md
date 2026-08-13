# Phase 05 Quantization Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows CPU 上为 Phase 04 的六个训练 run 构建可审计、可缓存、可失败隔离的 Q4_K_M GGUF 量化流水线，并通过唯一模型注册表与 llama-server manager 交付八个可验证模型（0.6B/1.7B 各 Base、SFT、DPO b01、DPO b03），同时保留两个 SFT F16 anchor。

**Architecture:** `src/slot_extractor/quantization/` 承载纯 Python 的 registry、lineage、manifest、缓存键和 pipeline 编排接口；`scripts/quantize/` 只负责 CLI 与外部 llama.cpp/LLaMA-Factory 命令适配。流水线严格按 `resolve → merge → convert-f16 → build-imatrix → quantize → verify` 执行，每个阶段写入原子 manifest，失败只污染该模型的 staging 目录，不影响其他矩阵项。`scripts/serve/` 使用同一注册表启动/停止 llama-server，沿用 `run_phase04_local.py` 的 Windows CPU 生命周期、健康检查和清理模式。

**Tech Stack:** Python 3.12、dataclasses/typing、JSON/YAML、SHA256、Windows subprocess/PowerShell、llama.cpp `llama-server`/`llama-quantize`/`llama-imatrix`/HF-to-GGUF converter、pytest、ruff、现有 `collect_artifacts.py` manifest 与 `run_phase04_local.py` subprocess 模式。

## Global Constraints

- **Matrix:** exactly 8 Q4_K_M targets: 0.6B/1.7B × Base,SFT,DPO b01,DPO b03; no extra quantization targets.
- **Anchors:** retain F16 anchors for `qwen3-0.6b-sft` and `qwen3-1.7b-sft`; anchors are verified outputs, not Q4 targets.
- **Pipeline:** every Q4 target must pass `resolve→merge→convert-f16→build-imatrix→quantize→verify` in this exact order; the two F16 anchors intentionally stop after `resolve→merge→convert-f16→verify`.
- **Calibration:** imatrix input must be a versioned domain calibration corpus and must never resolve to `data/eval/test.jsonl` or any path under `data/eval/`.
- **Registry:** `configs/quantization/phase05.yaml` is the one canonical model registry; evaluation, UI and serving may reference its model IDs but may not duplicate model metadata or model-to-path maps.
- **Lineage:** every artifact records base model/revision, adapter run IDs, parent artifact IDs, source SHA256, tool versions, command and git revision.
- **Integrity:** manifests contain SHA256 for inputs and outputs; writes are atomic and verification re-hashes files.
- **Cache:** cache keys include registry identity, lineage, source hashes, tool versions and stage parameters; cache hits still run verification.
- **Failure isolation:** each target/stage uses a separate staging directory and atomic promotion; a failed target is marked failed and does not block independent targets.
- **Platform:** Windows CPU only; subprocesses must clear CUDA visibility and use deterministic text encoding.
- **Serving:** llama-server manager consumes registry entries, binds localhost, checks `/v1/models`, and always terminates/cleans up processes.
- **Scope:** no threshold gate, no automatic Q5/Q8 fallback or escalation, and no product-code implementation in this plan.

---

## File structure

- Create: `src/slot_extractor/quantization/__init__.py` — public exports for the quantization domain.
- Create: `src/slot_extractor/quantization/registry.py` — typed loader for the sole canonical registry in `configs/quantization/phase05.yaml`, containing 8 Q4 targets plus 2 SFT F16 anchors and their output artifact paths.
- Create: `src/slot_extractor/quantization/lineage.py` — immutable lineage records and deterministic lineage/cache-key derivation.
- Create: `src/slot_extractor/quantization/manifest.py` — stage manifest schema, SHA256 helpers, atomic JSON writes, and verification.
- Create: `src/slot_extractor/quantization/pipeline.py` — resolve/merge/convert/imatrix/quantize/verify orchestration and failure isolation.
- Create: `src/slot_extractor/quantization/runner.py` — Windows CPU subprocess runner and tool discovery/version capture.
- Create: `scripts/quantize/run_phase05.py` — matrix CLI, resume/cache behavior, and failure summary.
- Create: `src/slot_extractor/inference/llama_server_manager.py` — registry-driven llama-server lifecycle manager reusable by serving and evaluation.
- Create: `configs/quantization/phase05.yaml` — canonical model registry plus paths, tool binaries, calibration data, exact matrix and anchor policy; every consumer reads this file.
- Modify: `scripts/train/collect_artifacts.py` — expose/consume stable Phase 04 manifest fields without changing existing collection semantics.
- Modify: `scripts/eval/run_phase04_local.py` — reuse the server manager interface for the Phase 05 local verification path.
- Modify: `scripts/serve/start_llama_server.ps1` — accept a registry model ID and delegate path/port/CPU options to the manager.
- Test: `tests/unit/quantization/test_registry.py`, `test_lineage.py`, `test_manifest.py`, `test_runner.py`, `test_pipeline.py`, `test_llama_server_manager.py` — deterministic unit coverage with fake tools.
- Test: `tests/integration/test_phase05_quantization.py` — fake end-to-end matrix and real artifact verification; no model downloads.

### Task 1: Define the canonical registry and exact matrix

**Files:**
- Create: `src/slot_extractor/quantization/__init__.py`
- Create: `src/slot_extractor/quantization/registry.py`
- Create: `configs/quantization/phase05.yaml`
- Test: `tests/unit/quantization/test_registry.py`

**Interfaces:**
- Produces `ModelSpec(model_id: str, family: str, size_b: float, stage: Literal["base","sft","dpo"], variant: str, base_model: str, base_revision: str, adapter_run_id: str | None, parent_model_id: str | None, artifact_kind: Literal["f16","q4_k_m"], artifact_path: Path, manifest_path: Path, is_anchor: bool)`.
- Produces `ModelRegistry.from_config(path: Path) -> ModelRegistry`, `.get(model_id: str) -> ModelSpec`, `.quantization_targets() -> tuple[ModelSpec, ...]`, `.anchors() -> tuple[ModelSpec, ...]`.

- [ ] **Step 1: Write the failing registry tests.**

```python

def test_registry_has_exactly_eight_q4_targets_and_two_sft_anchors():
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    assert len(registry.quantization_targets()) == 8
    assert {m.artifact_kind for m in registry.quantization_targets()} == {"q4_k_m"}
    assert {m.model_id for m in registry.anchors()} == {
        "qwen3-0.6b-sft-f16", "qwen3-1.7b-sft-f16"
    }


def test_registry_rejects_duplicate_model_ids():
    with pytest.raises(RegistryError, match="duplicate model_id"):
        ModelRegistry((ModelSpec("x", "qwen3", .6, "base", "", "m", "main", None, None, "f16", Path("x.gguf"), Path("x.manifest.json"), False),) * 2)
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `uv run pytest tests/unit/quantization/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slot_extractor.quantization'`.

- [ ] **Step 3: Implement the registry and config.** The config must enumerate artifact/manifest paths and the eight IDs `qwen3-{0.6b,1.7b}-{base,sft,dpo-b01,dpo-b03}-q4-k-m` and the two `qwen3-{0.6b,1.7b}-sft-f16` anchors; DPO records point to their matching Phase 04 SFT parent and adapter run.

```python
@dataclass(frozen=True)
class ModelRegistry:
    models: tuple[ModelSpec, ...]

    def __post_init__(self) -> None:
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise RegistryError("duplicate model_id")
        if len(self.quantization_targets()) != 8:
            raise RegistryError("Phase 05 requires exactly 8 Q4_K_M targets")

    def get(self, model_id: str) -> ModelSpec:
        try:
            return next(m for m in self.models if m.model_id == model_id)
        except StopIteration as exc:
            raise RegistryError(f"unknown model_id: {model_id}") from exc
```

- [ ] **Step 4: Run the tests and lint.**

Run: `uv run pytest tests/unit/quantization/test_registry.py -v && uv run ruff check src/slot_extractor/quantization tests/unit/quantization/test_registry.py`
Expected: all registry tests PASS and ruff reports no errors.

- [ ] **Step 5: Commit the registry contract.**

```bash
git add src/slot_extractor/quantization/__init__.py src/slot_extractor/quantization/registry.py configs/quantization/phase05.yaml tests/unit/quantization/test_registry.py
git commit -m "feat: define phase05 model registry"
```

### Task 2: Add explicit lineage, hashes, manifests, and cache keys

**Files:**
- Create: `src/slot_extractor/quantization/lineage.py`
- Create: `src/slot_extractor/quantization/manifest.py`
- Test: `tests/unit/quantization/test_lineage.py`, `tests/unit/quantization/test_manifest.py`
- Modify: `scripts/train/collect_artifacts.py:72-83` only if a stable `source_sha256`/`lineage` field is needed for compatibility.

**Interfaces:**
- Produces `Lineage(model_id: str, base_model: str, base_revision: str, parent_model_id: str | None, adapter_run_id: str | None, source_sha256: tuple[tuple[str,str], ...], git_revision: str, tool_versions: tuple[tuple[str,str], ...])`.
- Produces `StageManifest(model_id: str, stage: str, status: Literal["running","complete","failed"], artifact_kind: Literal["f16","q4_k_m"], is_anchor: bool, cache_key: str, lineage: Lineage, inputs: tuple[ArtifactHash,...], outputs: tuple[ArtifactHash,...], command: tuple[str,...], error: str | None)`.
- Produces `sha256_file(path: Path) -> str`, `write_manifest_atomic(path: Path, manifest: StageManifest) -> None`, `read_and_verify_manifest(path: Path) -> StageManifest` and `cache_key(lineage: Lineage, stage: str, parameters: Mapping[str,str]) -> str`.

- [ ] **Step 1: Write failing tests for deterministic lineage and tamper detection.**

```python

def test_cache_key_changes_when_source_hash_or_tool_version_changes():
    first = cache_key(lineage, "quantize", {"type": "Q4_K_M"})
    changed = replace(lineage, tool_versions=(("llama-quantize", "new"),))
    assert first != cache_key(changed, "quantize", {"type": "Q4_K_M"})


def test_manifest_verification_rejects_changed_output(tmp_path: Path):
    output = tmp_path / "model.gguf"; output.write_bytes(b"ok")
    manifest_path = tmp_path / "manifest.json"
    write_manifest_atomic(manifest_path, complete_manifest_for(output))
    output.write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="sha256 mismatch"):
        read_and_verify_manifest(manifest_path)
```

- [ ] **Step 2: Run the tests to verify failure.**

Run: `uv run pytest tests/unit/quantization/test_lineage.py tests/unit/quantization/test_manifest.py -v`
Expected: FAIL because the lineage/manifest modules and symbols do not exist.

- [ ] **Step 3: Implement canonical JSON hashing and atomic manifests.** Sort all mapping keys, use UTF-8, write `<path>.tmp`, flush/close, then `Path.replace`; never overwrite a completed manifest with a partial one.

```python
def cache_key(lineage: Lineage, stage: str, parameters: Mapping[str, str]) -> str:
    payload = {"lineage": asdict(lineage), "stage": stage, "parameters": dict(sorted(parameters.items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
```

- [ ] **Step 4: Run focused tests and full existing artifact tests.**

Run: `uv run pytest tests/unit/quantization/test_lineage.py tests/unit/quantization/test_manifest.py tests/unit/test_collect_training_artifacts.py -v`
Expected: all PASS; existing Phase 04 artifact behavior remains unchanged.

- [ ] **Step 5: Commit provenance and manifest primitives.**

```bash
git add src/slot_extractor/quantization/lineage.py src/slot_extractor/quantization/manifest.py tests/unit/quantization/test_lineage.py tests/unit/quantization/test_manifest.py scripts/train/collect_artifacts.py
git commit -m "feat: add quantization lineage manifests and cache keys"
```

### Task 3: Implement Windows CPU tool runner

**Files:**
- Create: `src/slot_extractor/quantization/runner.py`
- Test: `tests/unit/quantization/test_runner.py`

**Interfaces:**
- Produces `Toolchain(resolve: Path, merge: Path, convert_f16: Path, imatrix: Path, quantize: Path, server: Path)`.
- Produces `CommandRunner.run(argv: Sequence[str], cwd: Path, log_path: Path) -> CompletedProcess[str]` and `.version(executable: Path) -> str`.

- [ ] **Step 1: Write failing tests for CPU environment and error logs.**

```python

def test_runner_sets_cpu_environment(monkeypatch, tmp_path, fake_subprocess):
    runner.run(["fake-tool", "--version"], tmp_path, tmp_path / "run.log")
    assert fake_subprocess.env["CUDA_VISIBLE_DEVICES"] == ""
    assert fake_subprocess.env["OMP_NUM_THREADS"] == "8"


def test_runner_includes_command_and_exit_code_in_error(tmp_path):
    with pytest.raises(ToolRunnerError, match="exit code 7"):
        runner.run(["fake-tool", "bad"], tmp_path, tmp_path / "run.log")
```

- [ ] **Step 2: Run focused tests and confirm failure.**

Run: `uv run pytest tests/unit/quantization/test_runner.py -v`
Expected: FAIL with missing runner implementation.

- [ ] **Step 3: Implement `subprocess.run` with Windows-safe executable resolution.** Set `CUDA_VISIBLE_DEVICES=""`, preserve caller environment, set `text=True`, `encoding="utf-8"`, `check=False`, append stdout/stderr to the stage log, and raise `ToolRunnerError` on nonzero return.

- [ ] **Step 4: Run tests and lint.**

Run: `uv run pytest tests/unit/quantization/test_runner.py -v && uv run ruff check src/slot_extractor/quantization/runner.py tests/unit/quantization/test_runner.py`
Expected: PASS and no lint errors.

- [ ] **Step 5: Commit the runner.**

```bash
git add src/slot_extractor/quantization/runner.py tests/unit/quantization/test_runner.py
git commit -m "feat: add windows cpu quantization runner"
```

### Task 4: Build the resolve-to-verify pipeline with failure isolation

**Files:**
- Create: `src/slot_extractor/quantization/pipeline.py`
- Test: `tests/unit/quantization/test_pipeline.py`

**Interfaces:**
- Produces `QuantizationPipeline(registry: ModelRegistry, toolchain: Toolchain, runner: CommandRunner, paths: PipelinePaths)`.
- Produces `.run(model_id: str, *, force: bool = False) -> StageManifest` and `.run_matrix(*, continue_on_error: bool = True) -> MatrixResult`.
- Each stage method is private but exact: `_resolve`, `_merge`, `_convert_f16`, `_build_imatrix`, `_quantize`, `_verify`; each accepts `(spec: ModelSpec, work_dir: Path, manifest: StageManifest) -> StageManifest`.

- [ ] **Step 1: Write failing tests asserting exact stage order, cache hit verification, and isolated failure.**

```python

def test_pipeline_runs_exact_stage_order(fake_runner, registry, paths):
    result = QuantizationPipeline(registry, tools, fake_runner, paths).run("qwen3-0.6b-base-q4-k-m")
    assert fake_runner.commands == ["resolve", "merge", "convert-f16", "build-imatrix", "quantize", "verify"]
    assert result.status == "complete"


def test_failed_target_does_not_stop_matrix(fake_runner_that_fails_one_target, registry, paths):
    result = QuantizationPipeline(registry, tools, fake_runner_that_fails_one_target, paths).run_matrix()
    assert result.failed == ("qwen3-0.6b-base-q4-k-m",)
    assert len(result.completed) == 7
```

- [ ] **Step 2: Run the focused tests to verify failure.**

Run: `uv run pytest tests/unit/quantization/test_pipeline.py -v`
Expected: FAIL because `QuantizationPipeline` is undefined.

- [ ] **Step 3: Implement stage orchestration.** `resolve` reads Phase 04 manifests via `verify_artifacts(run_dir, training_only=True)` for SFT/DPO, verifies the manifest base model against registry lineage, resolves the adapter path under the run directory and rejects path traversal, or resolves the base model directly; `merge` applies an adapter only for SFT/DPO; `convert-f16` writes the HF-to-GGUF F16 artifact; `build-imatrix` uses only the configured domain calibration corpus and explicitly rejects `data/eval/test.jsonl` and every path under `data/eval/`; `quantize` invokes exactly `Q4_K_M`; `verify` checks GGUF existence, nonzero size, expected model ID metadata and SHA256. Never introduce a metric threshold or Q5/Q8 branch.

```python
STAGES = ("resolve", "merge", "convert-f16", "build-imatrix", "quantize", "verify")

for stage in STAGES:
    if cached_complete(stage_manifest, expected_key):
        read_and_verify_manifest(stage_manifest_path)
        continue
    stage_dir = work_root / spec.model_id / f".{stage}.staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = getattr(self, f"_{stage.replace('-', '_')}")(spec, stage_dir, manifest)
        write_manifest_atomic(stage_dir / "manifest.json", manifest)
        stage_dir.replace(final_stage_dir)
    except (OSError, ToolRunnerError, ManifestError) as exc:
        write_manifest_atomic(stage_dir / "manifest.json", replace(manifest, status="failed", error=str(exc)))
        raise
```

- [ ] **Step 4: Run tests including current Phase 04 tests.**

Run: `uv run pytest tests/unit/quantization/test_pipeline.py tests/unit/test_run_phase04_local.py tests/unit/test_collect_training_artifacts.py -v`

Add a test that configures calibration as `data/eval/test.jsonl` and asserts `PipelineError("calibration data must not use frozen evaluation data")` before any external tool runs.
Expected: all PASS; existing Phase 04 lifecycle behavior is unchanged.

- [ ] **Step 5: Commit the pipeline.**

```bash
git add src/slot_extractor/quantization/pipeline.py tests/unit/quantization/test_pipeline.py
git commit -m "feat: implement isolated phase05 quantization pipeline"
```

### Task 5: Add matrix CLI, resume behavior, and integration test

**Files:**
- Create: `scripts/quantize/run_phase05.py`
- Test: `tests/integration/test_phase05_quantization.py`

**Interfaces:**
- Produces `main(argv: list[str] | None = None) -> int` with `--config`, `--model-id` (repeatable), `--force`, `--continue-on-error`, and `--summary`.
- CLI returns `0` only when every requested target completes; returns `1` with a JSON failure summary when any target fails.

- [ ] **Step 1: Write a fake-tool integration test first.**

```python

def test_cli_builds_eight_targets_and_resumes_verified_cache(tmp_path, monkeypatch):
    assert main(["--config", str(config), "--continue-on-error"]) == 0
    assert len(list((tmp_path / "models/gguf").glob("*.gguf"))) == 8
    before_second_run = len(fake_runner.invocations)
    assert main(["--config", str(config)]) == 0
    assert len(fake_runner.invocations) == before_second_run  # cache hits re-hash manifests without rerunning tools
```

- [ ] **Step 2: Run the integration test and verify it fails.**

Run: `uv run pytest tests/integration/test_phase05_quantization.py -v`
Expected: FAIL because the CLI and pipeline integration do not exist.

- [ ] **Step 3: Implement CLI argument parsing and summary output.** Use `registry.quantization_targets()` by default, preserve per-model manifests under `models/quantization/<model-id>/`, and never delete another model's staging/output on failure.

- [ ] **Step 4: Run integration and CLI help checks.**

Run: `uv run pytest tests/integration/test_phase05_quantization.py -v && uv run python scripts/quantize/run_phase05.py --help`
Expected: PASS; help lists `--config`, `--model-id`, `--force`, `--continue-on-error`, and `--summary`.

- [ ] **Step 5: Commit the CLI.**

```bash
git add scripts/quantize/run_phase05.py tests/integration/test_phase05_quantization.py
git commit -m "feat: add phase05 quantization matrix cli"
```

### Task 6: Verify and retain the two SFT F16 anchors

**Files:**
- Modify: `src/slot_extractor/quantization/pipeline.py`
- Modify: `scripts/quantize/run_phase05.py`
- Test: `tests/unit/quantization/test_pipeline.py`

**Interfaces:**
- Adds `QuantizationPipeline.run_anchors() -> tuple[StageManifest, ...]` for exactly `qwen3-0.6b-sft-f16` and `qwen3-1.7b-sft-f16`.
- Anchor manifests have `artifact_kind="f16"`, `is_anchor=true`, and stop after verified F16 output; they do not build imatrix or Q4 output.

- [ ] **Step 1: Write the failing anchor tests.**

```python

def test_run_anchors_keeps_only_two_sft_f16_outputs(pipeline):
    manifests = pipeline.run_anchors()
    assert [m.model_id for m in manifests] == ["qwen3-0.6b-sft-f16", "qwen3-1.7b-sft-f16"]
    assert all(m.status == "complete" and m.artifact_kind == "f16" for m in manifests)
    assert "quantize" not in fake_runner.stage_names
```

- [ ] **Step 2: Run focused test and verify failure.**

Run: `uv run pytest tests/unit/quantization/test_pipeline.py -k anchor -v`
Expected: FAIL because `run_anchors` does not exist.

- [ ] **Step 3: Implement anchor resolution/merge/convert/verify and CLI `--anchors`.** Anchors are retained in `models/merged/<anchor-id>/` and are independently hashed; no automatic conversion to Q4, Q5, or Q8 is permitted.

- [ ] **Step 4: Run focused and full quantization tests.**

Run: `uv run pytest tests/unit/quantization tests/integration/test_phase05_quantization.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit anchor retention.**

```bash
git add src/slot_extractor/quantization/pipeline.py scripts/quantize/run_phase05.py tests/unit/quantization/test_pipeline.py
git commit -m "feat: retain phase05 sft f16 anchors"
```

### Task 7: Add registry-driven llama-server manager and Phase 04 compatibility

**Files:**
- Create: `src/slot_extractor/inference/llama_server_manager.py`
- Modify: `scripts/serve/start_llama_server.ps1`
- Modify: `scripts/eval/run_phase04_local.py:83-155`
- Test: `tests/unit/quantization/test_llama_server_manager.py`, `tests/unit/test_run_phase04_local.py`

**Interfaces:**
- Produces `LlamaServerManager(registry: ModelRegistry, server: Path, host: str = "127.0.0.1", port: int = 8080, threads: int = 8)` with public read-only `host` and `port` attributes.
- Produces `.start(model_id: str, log_path: Path) -> subprocess.Popen[str]`, `.wait_ready(process, timeout_s: float) -> None`, `.stop(process) -> None`, and `.base_url -> str` equal to `http://<host>:<port>/v1`.

- [ ] **Step 1: Write failing lifecycle tests.**

```python

def test_manager_uses_registry_path_and_cpu_flags(fake_popen, manager):
    process = manager.start("qwen3-0.6b-base-q4-k-m", tmp_path / "server.log")
    assert fake_popen.argv == ["llama-server.exe", "-m", str(registry.get("qwen3-0.6b-base-q4-k-m").artifact_path), "--host", "127.0.0.1", "--port", "8080", "--threads", "8"]
    assert fake_popen.env["CUDA_VISIBLE_DEVICES"] == ""


def test_manager_stops_process_after_timeout(fake_process, manager):
    with pytest.raises(ServerError):
        manager.wait_ready(fake_process, timeout_s=0.01)
    manager.stop(fake_process)
    fake_process.terminate.assert_called_once()
```

- [ ] **Step 2: Run tests to verify failure.**

Run: `uv run pytest tests/unit/quantization/test_llama_server_manager.py -v`
Expected: FAIL because the manager does not exist.

- [ ] **Step 3: Implement manager and PowerShell delegation.** The manager resolves `spec.artifact_path` and verifies `spec.manifest_path` through `read_and_verify_manifest()` only through the registry, sets `CUDA_VISIBLE_DEVICES=""`, polls `http://127.0.0.1:<port>/v1/models`, and guarantees terminate/kill in `finally`. Update `start_llama_server.ps1` to accept `-ModelId`, `-Port`, and `-Threads`; it must reject unknown IDs rather than accept an arbitrary file path.

- [ ] **Step 4: Re-run manager and existing local-evaluation tests.**

Run: `uv run pytest tests/unit/quantization/test_llama_server_manager.py tests/unit/test_run_phase04_local.py -v`
Expected: all PASS; Phase 04 tests retain their existing API behavior.

- [ ] **Step 5: Commit serving integration.**

```bash
git add src/slot_extractor/inference/llama_server_manager.py scripts/serve/start_llama_server.ps1 scripts/eval/run_phase04_local.py tests/unit/quantization/test_llama_server_manager.py tests/unit/test_run_phase04_local.py
git commit -m "feat: add registry driven llama server manager"
```

### Task 8: End-to-end verification, documentation, and final quality gate

**Files:**
- Modify: `docs/project-structure.md:53-56` — document the new quantization package and CLI paths.
- Modify: `deployment/llama_cpp/.gitkeep` only if the implementation needs a checked-in tool metadata placeholder; never commit binaries.
- Test: `tests/integration/test_phase05_quantization.py`

**Interfaces:**
- Final operator command: `uv run python scripts/quantize/run_phase05.py --config configs/quantization/phase05.yaml --anchors --continue-on-error --summary`.
- Final serving command: `powershell -ExecutionPolicy Bypass -File scripts/serve/start_llama_server.ps1 -ModelId qwen3-0.6b-sft-q4-k-m`.

- [ ] **Step 1: Add an end-to-end test for all requirements.** Assert eight Q4 manifests, two F16 anchor manifests, exact stage order, lineage fields, SHA256 verification, cache reuse, no Q5/Q8 commands, and one injected target failure leaving seven complete targets.

```python

def test_phase05_contract(fake_tools, tmp_path):
    result = run_fake_matrix(tmp_path, continue_on_error=True)
    assert len(result.completed) == 7
    assert result.failed == ("qwen3-0.6b-base-q4-k-m",)
    assert len(result.anchors) == 2
    assert all("Q4_K_M" in m.command for m in result.completed)
    assert all(m.lineage.base_revision for m in result.completed)
    assert all(read_and_verify_manifest(m.path).status == "complete" for m in result.completed)
    assert not any("Q5" in c or "Q8" in c for c in fake_tools.commands)
```

- [ ] **Step 2: Run the complete verification suite.**

Run: `uv run pytest -q`
Expected: all tests PASS, including all existing Phase 03/04 tests and new quantization tests.

Run: `uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Run CLI dry-run/help and inspect generated manifests.**

Run: `uv run python scripts/quantize/run_phase05.py --config configs/quantization/phase05.yaml --help`
Expected: command exits 0 and documents exact matrix/anchor semantics without threshold or fallback options.

Run: `uv run python scripts/quantize/run_phase05.py --config configs/quantization/phase05.yaml --anchors --continue-on-error --summary`
Expected on a machine without configured llama.cpp tools: exit 1, each unavailable target has an isolated `status=failed` manifest and a summary; no target is silently marked complete and no other target directory is deleted. Expected on a provisioned Windows CPU machine: exit 0, eight verified Q4 manifests and two verified F16 anchor manifests.

- [ ] **Step 4: Update the project structure documentation.** State that reusable quantization logic lives under `src/slot_extractor/quantization`, the CLI is under `scripts/quantize`, artifacts are under ignored `models/`, and the registry is the only model-to-path authority.

- [ ] **Step 5: Perform self-review before commit.** Check every global constraint against a task, search for forbidden placeholders and automatic fallback language, and confirm all signatures referenced by later tasks are defined earlier.

Run: `python -c "from pathlib import Path; text=Path('docs/superpowers/plans/2026-08-12-phase05-quantization-infrastructure.md').read_text(encoding='utf-8').split('## Self-review checklist',1)[0]; forbidden=['TBD','TODO','implement later','Similar to Task','add appropriate error handling']; print({x:text.count(x) for x in forbidden})"`
Expected: every placeholder count is `0`; the terms used by this self-review procedure are excluded from the checklist section itself.

- [ ] **Step 6: Commit final plan-adjacent documentation and tests.**

```bash
git add docs/project-structure.md tests/integration/test_phase05_quantization.py
git commit -m "docs: complete phase05 quantization infrastructure plan

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Self-review checklist

- **Spec coverage:** exact 8 Q4_K_M targets, unique registry, explicit lineage, ordered six-stage pipeline, manifests/SHA256/cache, isolated failures, Windows CPU, llama-server manager, two SFT F16 anchors, and explicit no-threshold/no-Q5-Q8 policy are covered by Global Constraints and Tasks 1–8.
- **Repository fit:** Phase 04 run IDs and manifests are consumed through `verify_artifacts`; subprocess and cleanup behavior follows `run_phase04_local.py`; evaluation/UI/serving consume the same `ModelRegistry.from_config(configs/quantization/phase05.yaml)` contract and scripts stay thin.
- **File structure:** every new implementation/test/config path is listed before tasks; no product code is requested outside the named interfaces.
- **Placeholder scan:** no `TBD`, `TODO`, “implement later”, or vague “add tests” steps; every code-changing step includes a concrete signature/snippet and exact command/Expected result.
- **Type consistency:** `ModelSpec`, `ModelRegistry`, `Lineage`, `StageManifest`, `Toolchain`, `CommandRunner`, `QuantizationPipeline`, and `LlamaServerManager` signatures are stable across tasks.
- **Execution scope:** this document is the only deliverable; model binaries, generated manifests, and real Phase 05 outputs are not committed by the plan author.
