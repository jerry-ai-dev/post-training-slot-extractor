from copy import deepcopy

import pytest
from test_raw_validator import _final

from slot_extractor.data.isolation import IsolationError, assert_no_eval_overlap, input_fingerprint


def test_fingerprint_ignores_json_key_order_and_whitespace() -> None:
    record = _final()
    reordered = deepcopy(record)
    reordered["input"] = dict(reversed(list(reordered["input"].items())))
    reordered["input"]["user_input"] = "  明天   "
    assert input_fingerprint(record) == input_fingerprint(reordered)


def test_overlap_is_rejected() -> None:
    train = _final()
    evaluation = {"id": "eval-1", "input": deepcopy(train["input"])}
    with pytest.raises(IsolationError, match="eval-1"):
        assert_no_eval_overlap([train], [evaluation])
