from test_raw_validator import _final

from slot_extractor.data.raw_sample import raw_sample_from_record
from slot_extractor.data.tag_audit import audit_tags


def test_audit_reports_category_deficit() -> None:
    report = audit_tags([raw_sample_from_record(_final())], {"追问": 0.60})
    assert report.deficits["追问"].required_ratio == 0.60


def test_hard_tag_satisfies_threshold() -> None:
    record = _final()
    record["tags"].append("相对时间")
    report = audit_tags([raw_sample_from_record(record)], {"追问": 0.50})
    assert not report.deficits
