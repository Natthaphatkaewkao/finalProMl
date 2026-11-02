import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import gpu_watchdog as wd  # noqa: E402


def test_decide_ok():
    assert wd.decide(temp_c=72, hot_s=0, run_s=600, on_ac=True, gpu_ok=True) is None


def test_decide_reasons_in_priority_order():
    assert wd.decide(None, 0, 0, True, gpu_ok=False) == "gpu_lost"
    assert wd.decide(90, 0, 0, True, True) == "temp_critical"
    assert wd.decide(86, 59, 0, True, True) is None
    assert wd.decide(86, 60, 0, True, True) == "temp_high"
    assert wd.decide(70, 0, 90 * 60, True, True) == "run_too_long"
    assert wd.decide(70, 0, 0, False, True) == "on_battery"


def test_cooldown_ready():
    assert not wd.cooldown_ready(temp_c=55, rest_s=20 * 60)
    assert not wd.cooldown_ready(temp_c=45, rest_s=10 * 60)
    assert wd.cooldown_ready(temp_c=50, rest_s=15 * 60)


def test_run_time_limit_can_be_disabled(monkeypatch):
    """ผู้ใช้สั่ง: พักตามอุณหภูมิเท่านั้น ไม่จำกัดเวลาทำงาน"""
    monkeypatch.setattr(wd, "MAX_RUN_SECONDS", None)
    assert wd.decide(70, 0, 10 * 3600, True, True) is None
    assert wd.decide(91, 0, 10 * 3600, True, True) == "temp_critical"


def test_parse_args_sets_limits():
    wd.apply_args(["--no-time-limit", "--min-rest-min", "0"])
    assert wd.MAX_RUN_SECONDS is None and wd.MIN_REST_SECONDS == 0
