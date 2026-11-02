import numpy as np
import optuna
import pandas as pd
import pytest

from tools import tuning_helpers as th

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_suggest_params_in_ranges():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(50):
        t = study.ask()
        p = th.suggest_params(t)
        study.tell(t, 0.0)
        assert 1e-4 <= p["lr_head"] <= 3e-3
        assert 1e-6 <= p["lr_ft"] <= 1e-4
        assert 0.1 <= p["unfreeze_frac"] <= 1.0
        assert p["head_epochs"] in range(1, 6)
        assert 0.0 <= p["dropout"] <= 0.5
        assert 1e-5 <= p["weight_decay"] <= 0.05
        assert 0.0 <= p["label_smoothing"] <= 0.15
        assert p["aug"] in ("none", "v1", "v2", "texture")
        assert p["mixup"] in (False, True)
        assert set(p) == set(th.DEFAULT_PARAMS)


def test_optuna_resumes_remaining_trials(tmp_path):
    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    s = optuna.create_study(study_name="x", storage=url, direction="maximize")
    s.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=2)
    s = optuna.load_study(study_name="x", storage=url)
    assert th.remaining_trials(s, 5) == 3
    s.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=th.remaining_trials(s, 5))
    assert len(s.trials) == 5
    assert th.remaining_trials(s, 5) == 0


def test_select_top1_tiebreak():
    df = pd.DataFrame({"candidate": ["a|none", "b|v1"], "val_macro_f1": [0.850, 0.845], "infer_ms_per_img": [30, 10]})
    assert th.select_top1(df) == "b|v1"
    df["val_macro_f1"] = [0.860, 0.845]
    assert th.select_top1(df) == "a|none"
    assert th.ROUNDS == ["none", "v1", "v2", "texture"] and th.DEFAULT_PARAMS["aug"] == "none"


def test_bootstrap_ci_bounds():
    y = np.array([0, 1, 2, 3] * 10)
    assert th.bootstrap_ci(y, y) == (1.0, 1.0)
    rng = np.random.default_rng(0)
    pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 4, len(y)))
    lo, hi = th.bootstrap_ci(y, pred)
    from sklearn.metrics import f1_score
    point = f1_score(y, pred, average="macro")
    assert lo <= point <= hi


def test_mcnemar_counts():
    y = np.array([0, 1, 2, 3, 0, 1])
    a = np.array([0, 1, 2, 3, 0, 0])
    b = np.array([0, 1, 1, 0, 0, 0])
    assert th.mcnemar(y, a, b) == (2, 0, pytest.approx(0.5))


def test_make_sampler_resume_gives_new_params(tmp_path):
    """review #2: สร้าง sampler ใหม่ตอน resume ต้องไม่สุ่มค่าซ้ำกับ trial เดิม"""
    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    obj = lambda t: t.suggest_float("a", 0, 1)
    s = optuna.create_study(study_name="x", storage=url, direction="maximize", sampler=th.make_sampler(0))
    s.optimize(obj, n_trials=3)
    s = optuna.load_study(study_name="x", storage=url)
    s.sampler = th.make_sampler(len(s.trials))
    s.optimize(obj, n_trials=3)
    vals = [t.params["a"] for t in s.trials]
    assert len(set(vals)) == 6
    assert th.make_sampler(0)._n_startup_trials == 5


def test_check_export():
    """review #4: fp16 ต่างเล็กน้อยต้องไม่หยุด notebook แต่ผลต่างจริงต้องจับได้"""
    a = np.array([[0.6, 0.4, 0, 0]] * 20)
    ok, msg = th.check_export(a, a + np.array([[-0.004, 0.004, 0, 0]]))
    assert ok
    b = a.copy()
    b[:2] = [0.4, 0.6, 0, 0]
    ok, msg = th.check_export(a, b)
    assert not ok and "argmax" in msg


def test_fail_stale_trials(tmp_path):
    """minor #5: trial ที่ค้างสถานะ RUNNING (kernel ตายกลางทาง) ต้องถูกปิดเป็น FAIL ก่อนรันต่อ"""
    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    s = optuna.create_study(study_name="x", storage=url, direction="maximize")
    s.optimize(lambda t: t.suggest_float("a", 0, 1), n_trials=2)
    s.ask()  # trial ที่เริ่มแล้วแต่ไม่จบ = RUNNING ค้าง
    assert th.fail_stale_trials(s) == 1
    states = [t.state for t in optuna.load_study(study_name="x", storage=url).trials]
    assert optuna.trial.TrialState.RUNNING not in states
    assert th.fail_stale_trials(s) == 0


def _hist(train_acc, val_acc):
    return [{"epoch": i + 1, "train_accuracy": t, "val_accuracy": v, "val_macro_f1": v}
            for i, (t, v) in enumerate(zip(train_acc, val_acc))]


def test_diagnose_fit():
    """R4: วินิจฉัย overfitting จากช่องว่าง train-val accuracy ที่ epoch ที่ดีที่สุด"""
    assert "ปกติ" in th.diagnose_fit(_hist([0.7, 0.8, 0.82], [0.7, 0.79, 0.80]))
    assert "เล็กน้อย" in th.diagnose_fit(_hist([0.8, 0.9, 0.95], [0.75, 0.82, 0.81]))
    msg = th.diagnose_fit(_hist([0.9, 0.99, 1.0], [0.7, 0.78, 0.75]))
    assert "สูง" in msg and "epoch 2" in msg


def test_rounds_is_list_for_pandas():
    """review C1: ROUNDS ต้องเป็น list (tuple ทำให้ df[ROUNDS] หา column ไม่เจอ)"""
    assert isinstance(th.ROUNDS, list)
    df = pd.DataFrame({"m": ["a"] * len(th.ROUNDS), "r": th.ROUNDS, "v": range(len(th.ROUNDS))})
    assert list(df.pivot(index="m", columns="r", values="v")[th.ROUNDS].columns) == th.ROUNDS


def test_select_top1_same_arch_prefers_f1_over_ms_noise():
    """review I1: รอบต่าง ๆ ของโมเดลเดียวกันเร็วเท่ากัน -> เลือกตัวที่ F1 สูงกว่า ไม่ใช่เศษ ms"""
    df = pd.DataFrame({"candidate": ["resnet50|none", "resnet50|v1"], "val_macro_f1": [0.850, 0.845],
                       "infer_ms_per_img": [14.4, 14.1]})
    assert th.select_top1(df) == "resnet50|none"
    df = pd.DataFrame({"candidate": ["convnext_tiny|none", "resnet50|v1"], "val_macro_f1": [0.850, 0.845],
                       "infer_ms_per_img": [30.0, 14.1]})
    assert th.select_top1(df) == "resnet50|v1"


def test_diagnose_fit_uses_eval_train_accuracy():
    """review I2: ใช้ train accuracy แบบ eval mode (ไม่มี augmentation) ถ้ามีให้"""
    h = _hist([0.6, 0.7, 0.72], [0.7, 0.79, 0.80])  # train acc บนภาพที่ augment ต่ำกว่าจริง
    assert "ปกติ" in th.diagnose_fit(h)
    assert "สูง" in th.diagnose_fit(h, train_acc_eval=0.99)


def test_wait_for_cool_pauses_until_resume_temp():
    """ผู้ใช้สั่ง: หยุดทันทีเมื่อการ์ดจอถึง 60°C แล้วเทรนต่อเมื่อเย็นลง (50°C)"""
    temps = iter([61, 58, 55, 52, 50, 70])
    slept = []
    paused = th.wait_for_cool(lambda: next(temps), pause_at=60, resume_at=50, sleep=slept.append, poll_s=5)
    assert paused == 3 * 5 and slept == [5, 5, 5]  # 58, 55, 52 > 50 -> พัก 3 รอบ แล้ว 50 -> เทรนต่อ
    assert th.wait_for_cool(lambda: 59, pause_at=60, resume_at=50, sleep=slept.append) == 0


def test_wait_for_cool_ignores_unreadable_temp():
    def boom():
        raise RuntimeError("nvml not available")
    assert th.wait_for_cool(boom, pause_at=60, resume_at=50, sleep=lambda s: None) == 0
