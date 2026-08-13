"""Build the real Phase 05 GGUF matrix from cached Qwen bases and Phase 04 adapters."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from slot_extractor.quantization.lineage import Lineage, cache_key
from slot_extractor.quantization.manifest import (
    ArtifactHash,
    StageManifest,
    sha256_file,
    write_manifest_atomic,
)
from slot_extractor.quantization.registry import ModelRegistry, ModelSpec


@dataclass(frozen=True)
class Tools:
    converter: Path
    imatrix: Path
    quantize: Path


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        output.write(f"$ {' '.join(command)}\n")
        output.flush()
        completed = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, text=True, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def cached_base(spec: ModelSpec) -> Path:
    try:
        return Path(
            snapshot_download(spec.base_model, revision=spec.base_revision, local_files_only=True)
        )
    except Exception as error:
        cache_name = f"models--{spec.base_model.replace('/', '--')}"
        snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
        candidates = [
            path
            for path in snapshots.iterdir()
            if path.joinpath("config.json").is_file()
            and any(path.glob("*.safetensors"))
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot resolve one complete weight snapshot for {spec.base_model}"
            ) from error
        return candidates[0]


def merge_adapter(spec: ModelSpec, base: Path, destination: Path) -> Path:
    if destination.joinpath("config.json").is_file():
        return destination
    adapter = Path("experiments/runs") / f"phase04-{spec.adapter_run_id}" / "adapter"
    if not adapter.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter missing: {adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
    )
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(destination)
    del merged, model
    return destination


def convert(source: Path, output: Path, tools: Tools, log: Path) -> None:
    if output.is_file() and output.stat().st_size:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    run(
        [
            str(Path(".venv/Scripts/python.exe")),
            str(tools.converter),
            str(source),
            "--outfile",
            str(temporary),
            "--outtype",
            "f16",
        ],
        log,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("converter did not produce a non-empty GGUF")
    temporary.replace(output)


def build_imatrix(f16: Path, output: Path, tools: Tools, log: Path) -> None:
    if output.is_file() and output.stat().st_size:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(tools.imatrix),
            "-m",
            str(f16),
            "-f",
            "data/calibration/phase05-v1.txt",
            "-o",
            str(output),
            "-t",
            "8",
            "-c",
            "512",
            "-b",
            "128",
        ],
        log,
    )


def quantize(f16: Path, imatrix: Path, output: Path, tools: Tools, log: Path) -> None:
    if output.is_file() and output.stat().st_size:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(tools.quantize),
            "--imatrix",
            str(imatrix),
            str(f16),
            str(output),
            "Q4_K_M",
            "8",
        ],
        log,
    )


def manifest_for(spec: ModelSpec, output: Path, command: tuple[str, ...]) -> StageManifest:
    lineage = Lineage(
        spec.model_id,
        spec.base_model,
        spec.base_revision,
        spec.parent_model_id,
        spec.adapter_run_id,
        (),
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        (("llama.cpp", "local-release"),),
    )
    return StageManifest(
        spec.model_id,
        "verify",
        "complete",
        spec.artifact_kind,
        spec.is_anchor,
        cache_key(lineage, "real-build", {"type": spec.artifact_kind}),
        lineage,
        (),
        (ArtifactHash(str(output), sha256_file(output)),),
        command,
        None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", action="append")
    args = parser.parse_args(argv)
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    tools = Tools(
        Path("deployment/llama_cpp/source/convert_hf_to_gguf.py"),
        Path("deployment/llama_cpp/bin/llama-imatrix.exe"),
        Path("deployment/llama_cpp/bin/llama-quantize.exe"),
    )
    targets = registry.quantization_targets()
    if args.model_id:
        targets = tuple(registry.get(model_id) for model_id in args.model_id)
    f16_by_source: dict[tuple[str, str | None], Path] = {}
    for spec in targets:
        work = Path("models/quantization") / spec.model_id
        log = work / "real-build.log"
        base = cached_base(spec)
        source = (
            base
            if spec.stage == "base"
            else merge_adapter(spec, base, Path("models/merged") / f"hf-{spec.model_id}")
        )
        key = (spec.base_model, spec.adapter_run_id)
        anchor_id = f"qwen3-{spec.size_b:g}b-sft-f16"
        if spec.stage == "sft":
            f16 = registry.get(anchor_id).artifact_path
        else:
            f16 = work / "model-f16.gguf"
        convert(source, f16, tools, log)
        f16_by_source[key] = f16
        imatrix = Path("models/imatrix") / f"{spec.model_id}.dat"
        build_imatrix(f16, imatrix, tools, log)
        quantize(f16, imatrix, spec.artifact_path, tools, log)
        write_manifest_atomic(
            spec.manifest_path, manifest_for(spec, spec.artifact_path, ("Q4_K_M",))
        )
        if spec.stage == "sft":
            anchor = registry.get(anchor_id)
            write_manifest_atomic(anchor.manifest_path, manifest_for(anchor, f16, ("convert-f16",)))
        print(f"complete: {spec.model_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
