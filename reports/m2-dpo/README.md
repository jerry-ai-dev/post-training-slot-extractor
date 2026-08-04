# M2 DPO comparison

Winner: `qwen3-1.7b-sft`

## SFT parents and DPO runs

| run_id | status | protocol | task_correctness | effective_pass | mean | p95 |
|---|---|---:|---:|---:|---:|---:|
| `qwen3-0.6b-sft` | evaluated | 82.4% | 77.3% | 24/51 | 45596ms | 53900ms |
| `qwen3-1.7b-sft` | evaluated | 88.2% | 92.1% | 29/51 | 95664ms | 110400ms |
| `qwen3-0.6b-dpo-b01` | evaluated | 80.4% | 75.2% | 22/51 | 38233ms | 44173ms |
| `qwen3-0.6b-dpo-b03` | evaluated | 80.4% | 76.5% | 22/51 | 38784ms | 44755ms |
| `qwen3-1.7b-dpo-b01` | evaluated | 88.2% | 91.9% | 29/51 | 85108ms | 98140ms |
| `qwen3-1.7b-dpo-b03` | evaluated | 88.2% | 89.8% | 27/51 | 85480ms | 98246ms |

## Selection rule application

Contenders within one effective pass: qwen3-1.7b-sft, qwen3-1.7b-dpo-b01.

Ineligible and failed runs remain in the table above.

## Parent diffs

- `qwen3-0.6b-sft` → `qwen3-0.6b-dpo-b01`: net effective pass -2
- `qwen3-0.6b-sft` → `qwen3-0.6b-dpo-b03`: net effective pass -2
- `qwen3-1.7b-sft` → `qwen3-1.7b-dpo-b01`: net effective pass +0
- `qwen3-1.7b-sft` → `qwen3-1.7b-dpo-b03`: net effective pass -2

> LLaMA-Factory CPU latency is not directly comparable to M0 llama.cpp latency.
