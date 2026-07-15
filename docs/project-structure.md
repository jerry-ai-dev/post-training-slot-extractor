# Project Structure

This project follows the spec workflow:

1. build inference harness and rule-based evaluation;
2. freeze evaluation data and establish M0 baseline;
3. build SFT / DPO training data;
4. run LLaMA-Factory SFT / DPO experiments;
5. merge, quantize to GGUF, and serve with llama.cpp;
6. analyze failures and iterate.

```text
.
├── configs/                       # Runtime and experiment configuration
│   ├── evaluation/                # Scoring thresholds, metric groups, report configs
│   ├── inference/                 # Backend configs for remote LLM, local llama-server, etc.
│   └── training/                  # LLaMA-Factory YAMLs and dataset registration
│       └── llamafactory/
│           ├── sft/
│           ├── dpo/
│           └── export/
├── data/                          # Versioned datasets and intermediate data
│   ├── raw/                       # Original logs, synthetic source drafts, manual seed cases
│   ├── interim/                   # Cleaned / transformed data before final split
│   ├── processed/                 # Train / val / DPO files ready for LLaMA-Factory
│   │   ├── sft/
│   │   ├── dpo/
│   │   └── llamafactory/
│   └── eval/                      # Frozen test.jsonl and holdout-like evaluation sets
├── deployment/                    # Local deployment assets
│   └── llama_cpp/                 # llama-server launch configs and notes
├── docs/                          # Engineering notes beyond the main spec
├── experiments/                   # Reproducible experiment outputs
│   ├── baselines/                 # M0 baseline outputs
│   └── runs/                      # SFT / DPO / quantization comparison runs
├── project-log/                   # Phase-by-phase construction logs
│   ├── phase-01-scaffold/
│   ├── phase-02-eval-baseline/
│   ├── phase-03-dataset/
│   ├── phase-04-training/
│   ├── phase-05-quantization-deploy/
│   ├── phase-06-iteration/
│   └── phase-07-report/
├── models/                        # Local model artifacts, ignored by git
│   ├── base/                      # Downloaded base HF models or references
│   ├── adapters/                  # LoRA / QLoRA adapters
│   ├── merged/                    # Merged fp16 / bf16 HF models
│   ├── gguf/                      # Quantized GGUF files
│   └── imatrix/                   # Importance matrix calibration outputs
├── reports/                       # Human-readable analysis and final project report
│   └── generated/                 # Generated scorecards and comparison tables
├── scripts/                       # Thin CLI entrypoints for common workflows
│   ├── data/                      # Dataset generation, split, validation commands
│   ├── eval/                      # Evaluation and scorecard commands
│   ├── train/                     # Local dry-run and cloud training helper commands
│   ├── quantize/                  # Merge, convert, imatrix, quantize commands
│   └── serve/                     # Local llama-server startup commands
├── src/
│   └── slot_extractor/            # Project Python package
│       ├── config/                # Typed config loading
│       ├── data/                  # Dataset builders, converters, validators
│       ├── evaluation/            # JSON parsing, schema checks, metrics, scorecards
│       ├── inference/             # Backend clients and shared inference harness
│       ├── prompts/               # Shared prompt templates for train / eval / deploy
│       ├── schemas/               # Output schema and assertion definitions
│       └── utils/                 # Shared utilities
└── tests/
    ├── fixtures/                  # Small deterministic sample cases
    ├── unit/                      # Unit tests for parsers, validators, scorers
    └── integration/               # End-to-end harness tests with mocked or local backends
```

## Naming Rules

- `data/eval/` is for frozen evaluation data. It must never be reused for training.
- `data/processed/` is for training-ready SFT / DPO data and LLaMA-Factory registration files.
- `experiments/` stores run outputs and metrics, while `reports/` stores curated summaries.
- `project-log/` stores construction logs by implementation phase. Each phase directory has a Markdown log for goals, tasks, decisions, commands, outputs, problems, and next steps.
- `models/` is intentionally ignored by git because it will contain large model artifacts.
- `scripts/` should stay thin; reusable logic belongs in `src/slot_extractor/`.
