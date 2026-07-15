# scripts/eval/collect_analysis.py
"""跑一个后端过全量评估集，落一份「逐样本明细」JSON 供离线分析。

与 slot-eval 的区别：slot-eval 只落聚合分数卡；本脚本额外记录每条样本的
输入、组装后的 messages、期望、模型原始输出、逐维度打分+detail、时延，
这样后续分析（如按维度抽样、归类错误）无需重新调用模型。

用法：
  uv run python scripts/eval/collect_analysis.py \
      --backend-config configs/inference/llama_server_qwen3_0.6b.yaml \
      --cases data/eval/test.jsonl \
      --out reports/analysis/qwen3-0.6b.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from slot_extractor.evaluation.runner import default_scorers
from slot_extractor.evaluation.scenarios import aggregate_scenario_slices
from slot_extractor.evaluation.scorecard import aggregate_scorecard, summarize_timing
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.results import CaseResult
from slot_extractor.schemas.sample import load_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect per-sample analysis log.")
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    samples = load_samples(Path(args.cases))
    backend = build_backend_from_config(Path(args.backend_config))
    prompt_builder = PromptBuilder()
    scorers = default_scorers()

    records = []
    case_results: list[CaseResult] = []
    for sample in samples:
        messages = prompt_builder.build_messages(sample)
        gen = backend.generate(messages)
        dims = {}
        for scorer in scorers:
            if scorer.applies_to(sample):
                ds = scorer.score(sample, gen)
                dimension_payload = {
                    "score": ds.score,
                    "passed": ds.passed,
                    "detail": ds.detail,
                }
                if scorer.dimension == "task_correctness":
                    dimension_payload.update(json.loads(ds.detail))
                dims[scorer.dimension] = dimension_payload
        case_results.append(
            CaseResult(
                sample_id=sample.id,
                layer=sample.layer,
                model_output=gen.text,
                dimensions={},
                total_ms=gen.total_ms,
                first_token_ms=gen.first_token_ms,
                tokens_per_s=gen.tokens_per_s,
            )
        )
        records.append(
            {
                "id": sample.id,
                "layer": sample.layer,
                "tags": sample.tags,
                "input": {
                    "history": sample.input.get("history"),
                    "current_state": sample.input.get("current_state"),
                    "user_input": sample.input.get("user_input"),
                    "current_time": sample.input.get("current_time"),
                    "available_tools": sample.input.get("available_tools"),
                },
                "reply_expectations": (
                    asdict(sample.reply_expectations) if sample.reply_expectations else None
                ),
                "messages_sent": messages,
                "expected": sample.expected,
                "model_output": gen.text,
                "dimensions": dims,
                "timing": {
                    "total_ms": gen.total_ms,
                    "first_token_ms": gen.first_token_ms,
                    "tokens_per_s": gen.tokens_per_s,
                },
            }
        )

    # 复用真实聚合逻辑，得到与分数卡一致的维度均分 + 时延统计
    from slot_extractor.schemas.results import DimensionScore

    scored_cases = []
    for cr, rec in zip(case_results, records, strict=True):
        cr_dims = {
            dim: DimensionScore(dim, d["score"], d["passed"], d["detail"])
            for dim, d in rec["dimensions"].items()
        }
        scored_cases.append(
            CaseResult(
                sample_id=cr.sample_id,
                layer=cr.layer,
                model_output=cr.model_output,
                dimensions=cr_dims,
                total_ms=cr.total_ms,
                first_token_ms=cr.first_token_ms,
                tokens_per_s=cr.tokens_per_s,
            )
        )

    scorecard = aggregate_scorecard(model=backend.model, cases=scored_cases)
    timing = summarize_timing(scored_cases)
    task_scores = {
        record["id"]: record["dimensions"]["task_correctness"]["score"]
        for record in records
    }

    payload = {
        "model": backend.model,
        "backend_config": str(args.backend_config),
        "cases_path": str(args.cases),
        "n": len(records),
        "aggregate_dimensions": {
            dim: {"score": ds.score, "passed": ds.passed, "detail": ds.detail}
            for dim, ds in scorecard.dimensions.items()
        },
        "aggregate_timing": {
            "count": timing.count,
            "total_ms_mean": timing.total_ms_mean,
            "total_ms_p50": timing.total_ms_p50,
            "total_ms_p95": timing.total_ms_p95,
            "total_ms_max": timing.total_ms_max,
            "total_ms_min": timing.total_ms_min,
            "first_token_ms_mean": timing.first_token_ms_mean,
            "tokens_per_s_mean": timing.tokens_per_s_mean,
        },
        "scenario_slices": aggregate_scenario_slices(samples, task_scores),
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(
        1 for r in records if r["dimensions"].get("protocol", {}).get("score") == 1.0
    )
    print(f"{backend.model}: wrote {out_path} (n={len(records)}, protocol_pass={passed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
