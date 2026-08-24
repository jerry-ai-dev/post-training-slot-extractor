"""Merge the final external LoRA and build the Phase 06 Round 006 GGUF matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class BuildError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BuildError(f"invalid config: {path}")
    return payload


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        output.write("$ " + " ".join(command) + "\n")
        output.flush()
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise BuildError(f"command failed ({completed.returncode}); see {log}")


def require_adapter(path: Path) -> None:
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise BuildError("external adapter is incomplete: " + ", ".join(missing))


def merge(base_model: str, adapter: Path, output: Path) -> None:
    if (output / "config.json").is_file() and any(output.glob("*.safetensors")):
        print(f"reuse merged model: {output}", flush=True)
        return
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output)


def build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/quantization/phase06-round006.yaml")
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-imatrix", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    adapter = args.adapter or Path(cfg["model"]["adapter_path"])
    require_adapter(adapter)
    paths = cfg["paths"]
    tools = cfg["tools"]
    merged = Path(paths["merged_hf"])
    f16 = Path(paths["f16_gguf"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / "build.log"
    merge(cfg["model"]["base_model"], adapter, merged)
    if not f16.is_file():
        f16.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                args.python,
                tools["converter"],
                str(merged),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ],
            log,
        )
    imatrix = output_dir / "r004-qwen3-0.6b-sft-imatrix.dat"
    iq_types = {item for item in cfg["quantizations"] if str(item).startswith("IQ")}
    if iq_types and not args.skip_imatrix and not imatrix.is_file():
        run(
            [
                tools["imatrix"],
                "-m",
                str(f16),
                "-f",
                paths["calibration_data"],
                "-o",
                str(imatrix),
                "-c",
                "512",
                "-b",
                "128",
            ],
            log,
        )
    artifacts: list[dict[str, Any]] = []
    for quant in cfg["quantizations"]:
        quant = str(quant)
        target = f16 if quant == "F16" else output_dir / f"r004-qwen3-0.6b-sft-{quant}.gguf"
        if quant != "F16" and not target.is_file():
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            command = [tools["quantize"]]
            if quant.startswith("IQ") and not args.skip_imatrix:
                command += ["--imatrix", str(imatrix)]
            command += [str(f16), str(temporary), quant]
            run(command, log)
            temporary.replace(target)
        artifacts.append(
            {"quantization": quant, "path": str(target), "bytes": target.stat().st_size}
        )
        print(f"ready: {quant} ({target})", flush=True)
    manifest = {
        "version": cfg["version"],
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": cfg["model"]["base_model"],
        "adapter": str(adapter),
        "artifacts": artifacts,
    }
    (output_dir / "matrix-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(build())
    except (BuildError, OSError, subprocess.SubprocessError) as error:
        print(f"phase06 round006 build error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
