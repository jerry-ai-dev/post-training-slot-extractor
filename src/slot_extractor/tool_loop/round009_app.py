from pathlib import Path

from fastapi import FastAPI

from slot_extractor.tool_loop.app import create_app

CONFIG = Path("configs/quantization/phase06-round009-app.yaml")


def create_round009_app() -> FastAPI:
    return create_app(
        quantization_config=CONFIG,
        log_path=Path("experiments/phase06/round-009/app/app.jsonl"),
        # Dedicated ports prevent this APP from silently connecting to the
        # legacy phase05 servers on 18080/18081.
        slot_ports={"left": 19080, "right": 19081},
        canonicalize_unique_matches=True,
    )
