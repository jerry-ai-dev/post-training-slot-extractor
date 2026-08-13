# Slot-Extractor Fine-tuning

Independent fine-tuning project for the appointment slot extractor described in `finetune-spec.md`.

Phase 05 includes a registry-driven quantization/evaluation pipeline and a local two-model
tool-loop comparison app. See `docs/project-structure.md` for repository ownership.

Start the comparison UI after provisioning the registered GGUF artifacts and llama.cpp binaries:

```powershell
uv run python -m uvicorn slot_extractor.tool_loop.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The app reads models from
`configs/quantization/phase05.yaml` and technicians from the versioned fixture; unavailable models
remain visible but cannot be selected.

## Reproducing Phase 05 models

The repository includes the six final LoRA adapters used by the Phase 05 experiment matrix under
`models/adapters/`. Each directory contains only `adapter_config.json` and
`adapter_model.safetensors`; training checkpoints and duplicated tokenizers are intentionally
excluded. The adapter configuration identifies its matching upstream Qwen base model.

Merged models and GGUF files are generated artifacts and are not committed. Rebuild them with the
registry and quantization pipeline:

```powershell
uv run python -m scripts.quantize.run_phase05 --config configs/quantization/phase05.yaml
```

The committed training and inference configurations, adapters, evaluation dataset, fixture, and
pipeline code are the reproducibility boundary. Base model downloads remain subject to their
upstream license and availability.
