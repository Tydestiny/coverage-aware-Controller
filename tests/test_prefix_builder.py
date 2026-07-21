from agent_error_compaction.adapters.aftraj import AFTrajAdapter
from agent_error_compaction.prefix_builder import build_prefixes


def failed():
    return AFTrajAdapter().convert_record({"conv_id": "bad", "turns": [{"content": "0"}, {"content": "1"}, {"content": "2"}], "mistake_step": 1})


def safe():
    return AFTrajAdapter().convert_record({"conv_id": "good", "turns": [{"content": "0"}, {"content": "1"}], "mistake_step": None})


def test_pre_error_prefix_is_false():
    prefix = build_prefixes([failed()], "all")[0]
    assert prefix.critical_error_occurred is False


def test_error_and_later_prefixes_are_true():
    prefixes = build_prefixes([failed()], "all")
    assert [item.critical_error_occurred for item in prefixes] == [False, True, True]


def test_safe_prefixes_are_always_false():
    assert not any(item.critical_error_occurred for item in build_prefixes([safe()], "all"))


def test_prefix_never_contains_future_steps():
    for prefix in build_prefixes([failed()], "all"):
        assert len(prefix.steps) == prefix.prefix_end_step + 1

