"""สร้าง tune_others.ipynb: จูน (ก่อน/Optuna/หลัง) ให้โมเดลที่ไม่ใช่ Top 1 เพื่อทำกราฟก่อน/หลังครบทุกโมเดล

ใช้ cell ตั้งค่า + โหลดข้อมูล + ฟังก์ชันเทรนจาก fabric_classification.ipynb (ไม่รัน EDA / ส่วน Top 1)
รันใต้ watchdog: .venv\\Scripts\\python.exe -I tools\\gpu_watchdog.py --no-time-limit --notebook tune_others.ipynb
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fabric_classification.ipynb"
OUT = ROOT / "tune_others.ipynb"
# config, imports, piece_id, manifest, clean, split, check, tuning_helpers, TRAIN
SETUP_PREFIXES = ["# ===== Config", "import sys, os, json", "def piece_id", 'MANIFEST_PATH = SPLIT_DIR',
                  "df = manifest.copy()", 'SPLIT_PATH = SPLIT_DIR', "# ตรวจข้อมูลก่อนเทรน",
                  '"""ฟังก์ชันช่วยของ notebook ส่วน 6', 'RUNS_CSV = RESULTS_DIR']

INTRO = """
# จูนโมเดลที่เหลือ (DenseNet121, EfficientNet-B0)

notebook หลักจูนเฉพาะ Top 1 (ResNet50) ส่วน ConvNeXt-Tiny ถูกจูนไว้ในรอบรันก่อนหน้า
notebook นี้ทำขั้นตอนเดียวกันกับอีก 2 โมเดล เพื่อให้มีกราฟก่อน/หลังจูนครบทุกโมเดล:
ใช้รอบ augmentation ที่ดีที่สุดของโมเดลนั้นจากรอบคัดเลือก → ก่อนจูน 3 seed → Optuna 15 trial → หลังจูน 3 seed
ผลเขียนที่ `results/others/` (ไม่ทับผลของ notebook หลัก)
"""

RUN = r'''
OTHERS = ["densenet121", "efficientnet_b0"]
OUT_DIR = RESULTS_DIR / "others"
OUT_DIR.mkdir(exist_ok=True)
screening = pd.read_csv(OUTPUT_DIR / "results" / "screening.csv")  # รอบคัดเลือกจาก notebook หลัก
storage = f"sqlite:///{(RESULTS_DIR / 'optuna.db').as_posix()}"
summary = []

for ARCH in OTHERS:
    sc = screening[screening.arch == ARCH]
    ROUND = sc.sort_values("val_macro_f1", ascending=False)["round"].iloc[0]
    BASE = {**DEFAULT_PARAMS, "aug": ROUND}
    name = fl.TV_ZOO[ARCH]["name"]
    print(f"\n===== {name}: รอบที่ดีที่สุด = {ROUND} =====")

    before = [train_one(ARCH, BASE, s, MAX_EPOCHS, "before") for s in SEEDS]

    study = optuna.create_study(study_name=f"other_{ARCH}_{ROUND}_ss2", storage=storage, direction="maximize",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3))
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    study.sampler = make_sampler(len(study.trials), SEED)

    def objective(trial):
        params = suggest_params(trial)
        try:
            rid = train_one(ARCH, params, SEED, TRIAL_EPOCHS, "trial", trial=trial)
        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        trial.set_user_attr("run_id", rid)
        return load_run(rid)["val_metrics"]["macro_f1"]

    fail_stale_trials(study)
    n_left = remaining_trials(study, N_TRIALS)
    print(f"Optuna {name}: มี {len(study.trials)} trial แล้ว จะรันเพิ่ม {n_left}")
    if n_left:
        study.optimize(objective, n_trials=n_left, catch=(Exception,), gc_after_trial=True)
    tr = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration", "user_attrs"))
    tr["duration_min"] = tr.pop("duration").dt.total_seconds() / 60
    tr.to_csv(OUT_DIR / f"optuna_trials_{ARCH}.csv", index=False)
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        print("คำนวณ importance ไม่ได้:", e)
        imp = {}
    pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(
        OUT_DIR / f"optuna_importance_{ARCH}.csv", index=False)
    best = {**BASE, **study.best_params}
    print(f"trial ที่ดีที่สุด #{study.best_trial.number}: Val Macro-F1 {study.best_value:.3f}")

    after = [train_one(ARCH, best, s, MAX_EPOCHS, "after") for s in SEEDS]

    for v, rids in (("before", before), ("after", after)):
        for s, rid in zip(SEEDS, rids):
            va = load_run(rid)["val_metrics"]
            te = evaluate(y_te, predict_split(rid, "test"))
            summary.append({"model": name, "arch": ARCH, "round": ROUND, "variant": v, "seed": s, "run_id": rid,
                            "val_macro_f1": va["macro_f1"], "val_accuracy": va["accuracy"],
                            "test_accuracy": te["accuracy"], "test_macro_f1": te["macro_f1"],
                            "params": json.dumps(best if v == "after" else BASE)})
    pd.DataFrame(summary).to_csv(OUT_DIR / "before_after_others.csv", index=False)
    plot_history({f"before s{s}": r for s, r in zip(SEEDS, before)} | {f"after s{s}": r for s, r in zip(SEEDS, after)},
                 name, f"before_after_curves_{ARCH}.png")

res = pd.DataFrame(summary)
display(res.groupby(["model", "variant"])[["val_macro_f1", "test_macro_f1"]].agg(["mean", "std"]).round(3))
'''


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    cells = [nbf.v4.new_markdown_cell(INTRO.strip())]
    for p in SETUP_PREFIXES:
        hit = [c for c in src["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith(p)]
        assert len(hit) == 1, f"หา cell ที่ขึ้นต้นด้วย {p!r} ไม่เจอ/เจอหลายอัน ({len(hit)})"
        cells.append(nbf.v4.new_code_cell("".join(hit[0]["source"])))
    cells.append(nbf.v4.new_code_cell(RUN.strip()))
    nb = nbf.v4.new_notebook(cells=cells, metadata=src.get("metadata", {}))
    nbf.write(nb, OUT)
    print("wrote", OUT, len(cells), "cells")


if __name__ == "__main__":
    main()
