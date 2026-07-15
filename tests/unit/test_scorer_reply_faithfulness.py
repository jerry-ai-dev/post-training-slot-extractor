from slot_extractor.evaluation.scorers.reply_faithfulness import _mentioned_technicians


def test_generic_gender_phrase_is_not_treated_as_technician_name() -> None:
    assert _mentioned_technicians("暂时没有符合条件的女技师。") == set()
    assert _mentioned_technicians("没有符合条件的泰式女技师。") == set()
    assert (
        _mentioned_technicians(
            "明天晚上8点没有符合条件的泰式女技师，您愿意调整时间、按摩类型或技师性别吗？"
        )
        == set()
    )
