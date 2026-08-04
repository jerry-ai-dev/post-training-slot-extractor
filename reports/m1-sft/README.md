# M1 SFT comparison

## M0 baseline

| M0 model | protocol | task_correctness | effective_pass |
|---|---:|---:|---:|
| `qwen3-0.6b` | 39.2% | 37.7% | 2/51 |
| `qwen3-1.7b` | 72.5% | 64.7% | 6/51 |
| `qwen3-4b-instruct-2507` | 82.4% | 67.6% | 25/51 |
| `gpt-5.6-sol` | 100.0% | 98.8% | 51/51 |

## SFT runs

| run_id | status | protocol | task_correctness | effective_pass | mean | p95 |
|---|---|---:|---:|---:|---:|---:|
| `qwen3-0.6b-sft` | evaluated | 82.4% | 77.3% | 24/51 | 45596ms | 53900ms |
| `qwen3-1.7b-sft` | evaluated | 88.2% | 92.1% | 29/51 | 95664ms | 110400ms |

> LLaMA-Factory CPU latency is not directly comparable to M0 llama.cpp latency.
