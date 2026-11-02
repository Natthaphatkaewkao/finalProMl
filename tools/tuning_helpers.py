"""ฟังก์ชันช่วยของ notebook ส่วน 6–10 (ถูกฝังลง notebook โดย tools/build_notebook.py)"""
from __future__ import annotations

import numpy as np

# รอบเทรนตามสไลด์: 1 ไม่ทำ augmentation, 2 augmentation แบบเบา (v1), 3 แบบหนัก (v2)
# + รอบ 4 texture: augmentation ที่รักษาพื้นผิวผ้า (ไม่มี blur / noise / หมุนเอียง)
ROUNDS = ["none", "v1", "v2", "texture"]  # list (ไม่ใช่ tuple) เพื่อใช้ df[ROUNDS] ได้
# สูตร default ของรอบคัดเลือก (aug ถูกแทนด้วยรอบนั้น ๆ)
DEFAULT_PARAMS = dict(lr_head=1e-3, lr_ft=1e-5, unfreeze_frac=0.3, head_epochs=3, dropout=0.3,
                      weight_decay=1e-4, label_smoothing=0.0, aug="none", mixup=False)


def suggest_params(trial) -> dict:
    """search space ของ Optuna (ตาม spec)"""
    return dict(
        lr_head=trial.suggest_float("lr_head", 1e-4, 3e-3, log=True),
        lr_ft=trial.suggest_float("lr_ft", 1e-6, 1e-4, log=True),
        unfreeze_frac=trial.suggest_float("unfreeze_frac", 0.1, 1.0),
        head_epochs=trial.suggest_int("head_epochs", 1, 5),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 0.05, log=True),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.15),
        aug=trial.suggest_categorical("aug", list(ROUNDS)),
        mixup=trial.suggest_categorical("mixup", [False, True]),
    )


def make_sampler(n_existing: int, seed: int = 42):
    """TPE ที่ seed ขึ้นกับจำนวน trial ที่มีอยู่แล้ว: รันต่อหลังหยุดกลางทางจะไม่สุ่มค่าซ้ำกับ trial เดิม
    n_startup_trials=5 (สุ่ม 5 trial แรก ที่เหลือให้ TPE เลือกจากผลก่อนหน้า)"""
    import optuna

    return optuna.samplers.TPESampler(seed=seed + n_existing, n_startup_trials=5)


def check_export(p_app, p_nb, max_mismatch: int = 1, warn_diff: float = 1e-2) -> tuple[bool, str]:
    """เทียบ probability จากแอป (load_predictor) กับ notebook: fp16 + batch ต่างกันทำให้ต่างเล็กน้อยได้
    ไม่ผ่านเมื่อคลาสที่ทายต่างกันเกิน max_mismatch รูป"""
    p_app, p_nb = np.asarray(p_app), np.asarray(p_nb)
    diff = float(np.abs(p_app - p_nb).max())
    mism = int((p_app.argmax(1) != p_nb.argmax(1)).sum())
    msg = f"max diff {diff:.1e}, argmax ต่างกัน {mism}/{len(p_app)} รูป"
    if mism > max_mismatch:
        return False, msg
    if diff > warn_diff:
        msg += " (ต่างมากกว่าปกติ ควรตรวจ preprocessing)"
    return True, msg


def wait_for_cool(read_temp, pause_at: float = 60, resume_at: float = 50, sleep=None, poll_s: float = 5) -> float:
    """ถ้าการ์ดจอร้อนถึง pause_at °C ให้หยุดรอจนเย็นลงถึง resume_at °C (คืนจำนวนวินาทีที่พัก)
    อ่านอุณหภูมิไม่ได้ (เช่นไม่มี GPU) -> ไม่พัก"""
    import time

    sleep = sleep or time.sleep
    try:
        if read_temp() < pause_at:
            return 0
        waited = 0
        while read_temp() > resume_at:
            sleep(poll_s)
            waited += poll_s
        return waited
    except Exception:
        return 0


def fail_stale_trials(study) -> int:
    """trial ที่ค้างสถานะ RUNNING (kernel ตาย/ถูกหยุดกลาง trial) -> FAIL เพื่อไม่ให้ค้างในผลและใน TPE"""
    import optuna

    stale = [t.number for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for n in stale:
        study.tell(n, state=optuna.trial.TrialState.FAIL)
    return len(stale)


def remaining_trials(study, n_total: int) -> int:
    """จำนวน trial ที่ยังต้องรัน (นับ trial ที่จบแล้วทุกสถานะ) ใช้ต่อการจูนที่ค้างไว้"""
    import optuna

    done = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL}
    return max(0, n_total - sum(t.state in done for t in study.trials))


def select_top1(df, tol: float = 0.01) -> str:
    """Val Macro-F1 สูงสุด; ถ้าตัวอื่นต่างไม่เกิน tol เลือกตัวที่ inference เร็วที่สุดในกลุ่มนั้น
    คืนค่า candidate = "<arch>|<round>" """
    best = df["val_macro_f1"].max()
    close = df[df["val_macro_f1"] >= best - tol - 1e-12].copy()
    # ความเร็วเป็นคุณสมบัติของสถาปัตยกรรม: ใช้ค่ามัธยฐานต่อ arch ปัดเป็น ms เต็ม เพื่อไม่ให้เศษ ms ตัดสิน
    arch = df["candidate"].str.split("|").str[0]
    speed = df.assign(arch=arch).groupby("arch")["infer_ms_per_img"].median().round()
    close["speed"] = close["candidate"].str.split("|").str[0].map(speed)
    return str(close.sort_values(["speed", "val_macro_f1"], ascending=[True, False])["candidate"].iloc[0])


def diagnose_fit(history, train_acc_eval: float | None = None) -> str:
    """วินิจฉัย overfitting จากช่องว่าง train−val accuracy ที่ epoch ที่ Val Macro-F1 ดีที่สุด
    train_acc_eval = accuracy ของชุด Train แบบ eval mode ไม่มี augmentation (ถ้ามีให้ใช้ค่านี้ เพราะ train accuracy
    ระหว่างเทรนวัดบนภาพที่ผ่าน augmentation/MixUp แล้ว จึงต่ำกว่าจริง)"""
    best = max(history, key=lambda h: h["val_macro_f1"])
    train_acc = best["train_accuracy"] if train_acc_eval is None else train_acc_eval
    gap = 100 * (train_acc - best["val_accuracy"])
    level = "ปกติ" if gap < 5 else ("overfitting เล็กน้อย" if gap <= 15 else "overfitting สูง")
    return f"{level} | best epoch {best['epoch']} | train−val accuracy gap {gap:.1f} จุด"


def bootstrap_ci(y, pred, B: int = 1000, seed: int = 42, n_classes: int = 4) -> tuple[float, float]:
    """ช่วงความเชื่อมั่น 95% ของ Macro-F1 ด้วย bootstrap (สุ่มรูปซ้ำแบบใส่คืน)"""
    from sklearn.metrics import f1_score

    y, pred = np.asarray(y), np.asarray(pred)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(B, len(y)))
    f1s = [f1_score(y[i], pred[i], labels=range(n_classes), average="macro", zero_division=0) for i in idx]
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b) -> tuple[int, int, float]:
    """b = a ถูก b ผิด, c = a ผิด b ถูก, p = exact binomial test (two-sided)"""
    from scipy.stats import binomtest

    ok_a, ok_b = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return b, c, float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
