from scripts.data.build_dataset import _generation_requests, _load_config


def test_generation_requests_cycle_scenarios_to_requested_counts() -> None:
    config = {
        "counts": {"追问": 7, "工具调用": 3},
        "scenarios": {
            "追问": ["ask_missing_time", "ask_missing_duration"],
            "工具调用": ["tool_general", "tool_specific"],
        },
    }
    requests = _generation_requests(config)
    assert len(requests) == 10
    assert [request.count for request in requests[:7]] == list(range(1, 8))
    assert [request.scenario for request in requests[:4]] == [
        "ask_missing_time",
        "ask_missing_duration",
        "ask_missing_time",
        "ask_missing_duration",
    ]


def test_production_config_requests_500_sft_and_150_dpo() -> None:
    config = _load_config("configs/data/phase03.yaml")
    assert len(_generation_requests(config)) == 500
    assert sum(config["dpo_target_counts"].values()) == 150
    assert config["output_root"] == "data"
    assert config["generation_concurrency"] == 5
