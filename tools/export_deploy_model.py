"""เปลี่ยนโมเดลที่แอปใช้ เป็น run ที่จูนแล้วตัวใดก็ได้ (ค่าเริ่มต้น: EfficientNet-B0 หลังจูน seed 42)

ทำอะไร:
1. สำรอง models/best_model.pth, model_meta.json, unseen_*.csv เดิมไว้ที่ backup_<วันที่>/ (ครั้งแรกเท่านั้น)
2. คัดลอก weights ของ run -> models/best_model.pth และเขียน model_meta.json (รูปแบบเดียวกับ notebook ส่วนที่ 12)
3. วัดผล Test / Unseen ของโมเดลใหม่ + ผล Unseen ของโมเดลหลังจูนทั้ง 4 ตัว
4. เขียน results/others/confusion_4models.json สำหรับหน้า "ก่อน vs หลังจูน" ของแอป
5. ตรวจว่าโหลดผ่าน fl.load_predictor (ทางเดียวกับแอป) แล้วผลตรงกับการโหลด weights ตรง ๆ

หมายเหตุ: ถ้ารัน fabric_classification.ipynb ใหม่ ส่วนที่ 12 จะ export ResNet50 ทับ ให้รันสคริปต์นี้อีกครั้ง
รัน: .venv\\Scripts\\python.exe -I tools\\export_deploy_model.py [run_id]
ต้องรัน tools/plot_before_after_4models.py ก่อน (สร้าง results/others/summary_4models.csv)
"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402

import fabric_lib as fl  # noqa: E402
from tuning_helpers import check_export  # noqa: E402

R, M = ROOT / "results", ROOT / "models"
DEFAULT_RUN = "ac7f583c51bb"  # EfficientNet-B0 หลังจูน seed 42 (Val ดีที่สุดใน 3 seed)
SELECTION_NOTE = ("เลือกหลังจูนครบ 4 โมเดล: Val Macro-F1 ต่างจากตัวที่สูงสุด (ConvNeXt-Tiny) ประมาณ 0.01 "
                  "จึงใช้เกณฑ์เดิม (ต่างไม่เกิน 0.01 เลือกตัวที่เบา/เร็วกว่า) ร่วมกับ Test ที่สูงและนิ่งที่สุด (3 seed) "
                  "และขนาดเล็กเหมาะกับ CPU · หมายเหตุ: ตัดสินใจหลังเห็นผล Test ของทุกโมเดลแล้ว")


def metrics(y, proba):
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    m = {"accuracy": accuracy_score(y, pred), "macro_precision": p.mean(), "macro_recall": r.mean(),
         "macro_f1": f.mean(), "min_recall": r.min()}
    for i, c in enumerate(fl.CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = p[i], r[i], f[i]
    return {k: float(v) for k, v in m.items()}


def ce(y, proba):
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-7, 1))))


def load(rid, device):
    h = json.loads((R / "histories" / f"{rid}.json").read_text(encoding="utf-8"))
    m = fl.build_model(h["arch"], dropout=h["params"]["dropout"], pretrained=False)
    m.load_state_dict(torch.load(M / "runs" / f"{rid}.pth", map_location="cpu", weights_only=True))
    return fl.TorchPredictor(m, device), h


def ms_per_img(pred, paths):
    pred.predict_proba(paths[:2])
    t = []
    for p in paths:
        t0 = time.perf_counter()
        pred.predict_proba([p])
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def main(rid=DEFAULT_RUN):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = {c: i for i, c in enumerate(fl.CLASSES)}
    summ = pd.read_csv(R / "others" / "summary_4models.csv")
    row = summ[summ.run_id == rid].iloc[0]

    # 1) สำรองของเดิม
    bk = ROOT / f"backup_{time.strftime('%Y-%m-%d')}"
    if not bk.exists():
        (bk / "models").mkdir(parents=True)
        (bk / "results").mkdir()
        for f in ("best_model.pth", "model_meta.json"):
            shutil.copy2(M / f, bk / "models" / f)
        for f in ("unseen_predictions.csv", "unseen_metrics.csv"):
            shutil.copy2(R / f, bk / "results" / f)
        print("สำรองของเดิมที่", bk)

    # 2) ข้อมูล Test / Val / Unseen (Val/Test 1 รูปต่อชิ้นผ้า เหมือน notebook)
    sp = pd.read_csv(R / "split.csv")
    ev = sp[sp.split != "train"]
    one = ev.sort_values("path").groupby("sample_id").sample(1, random_state=42)
    te = one[one.split == "test"].sort_values("path")
    va = one[one.split == "val"].sort_values("path")
    te_paths, y_te = [ROOT / "dataset" / p for p in te.path], te["class"].map(idx).to_numpy()
    va_paths = [ROOT / "dataset" / p for p in va.path]
    un = pd.read_csv(ROOT / "unseen_data" / "sources.csv")
    un_paths, y_un = [ROOT / "unseen_data" / f for f in un.file], un["class"].map(idx).to_numpy()
    assert len(te) == 328 and len(va) == 326

    pred, h = load(rid, dev)
    p_te, p_un = pred.predict_proba(te_paths), pred.predict_proba(un_paths)
    tm = metrics(y_te, p_te)
    cache = json.loads((R / "others" / "test_pred_cache.json").read_text())
    assert np.array_equal(p_te.argmax(1), np.array(cache[rid])), "ผลทาย Test ไม่ตรงกับตอนทำกราฟ"

    # 3) export
    shutil.copyfile(M / "runs" / f"{rid}.pth", M / "best_model.pth")
    name = row.model
    d = summ[summ.model == name]
    tba = {v: {"test_macro_f1_mean": float(g.test_macro_f1.mean()), "test_macro_f1_std": float(g.test_macro_f1.std(ddof=0)),
               "test_accuracy_mean": float(g.test_accuracy.mean())} for v, g in d.groupby("variant")}
    meta = {"framework": "torch", "model_file": "best_model.pth", "arch": h["arch"], "round": h["params"]["aug"],
            "model_name": f"{name} (aug {h['params']['aug']}, fine-tuned + Optuna)", "classes": fl.CLASSES,
            "img_size": fl.IMG_SIZE,
            "preprocessing": "RGB, EXIF-rotated, resize shorter side to 256, center crop 224, ImageNet mean/std",
            "params": h["params"], "seed": int(h["seed"]), "run_id": rid, "val_metrics": h["val_metrics"],
            "test_metrics": tm, "test_before_after": tba, "conf_threshold": 0.6,
            "infer_ms_per_img": ms_per_img(pred, va_paths[:30]),
            "infer_ms_per_img_cpu": ms_per_img(load(rid, "cpu")[0], va_paths[:30]),  # สำเนาแยก ไม่ย้ายโมเดล GPU
            "n_params": int(h["n_params"]), "fallback_needed": bool(tm["macro_f1"] < 0.85),
            "n_images": {"train": 4887, "val": 326, "test": 328}, "data_fingerprint": h["data_fingerprint"],
            "selection_note": SELECTION_NOTE, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (M / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ตรวจว่าโหลดแบบแอปแล้วได้ผลเดียวกัน
    app_pred = fl.load_predictor(M, dev)
    ok, msg = check_export(app_pred.predict_proba(va_paths[:20]), pred.predict_proba(va_paths[:20]))
    assert ok, msg
    print(f"Export {meta['model_name']} seed {meta['seed']}: Test acc {tm['accuracy']:.3f} F1 {tm['macro_f1']:.3f} | "
          f"GPU {meta['infer_ms_per_img']:.1f} ms, CPU {meta['infer_ms_per_img_cpu']:.1f} ms/รูป | load_predictor: {msg}")

    # 4) Unseen ของโมเดลในแอป + โมเดลหลังจูนทั้ง 4 ตัว (seed ที่ Val ดีที่สุด)
    up = un[["file", "class"]].assign(pred=[fl.CLASSES[i] for i in p_un.argmax(1)], conf=p_un.max(1),
                                      **{f"p_{c}": p_un[:, i] for i, c in enumerate(fl.CLASSES)})
    up.to_csv(R / "unseen_predictions.csv", index=False)
    um = pd.read_csv(R / "unseen_metrics.csv")
    um = um[~um.run.str.contains(r"tuned|\| (?:before|after) \(")]  # แทนแถวก่อน/หลังจูนเดิมด้วยแถว tuned
    rows, conf = [], {}
    for mo, g in summ.groupby("model", sort=False):
        conf[mo] = {}
        for v in ("before", "after"):
            t = g[g.variant == v].sort_values("val_macro_f1", ascending=False).iloc[0]
            cm = confusion_matrix(y_te, cache[t.run_id], labels=range(4))
            conf[mo][v] = {"seed": int(t.seed), "run_id": t.run_id, "matrix": cm.tolist(),
                           "test_macro_f1": float(t.test_macro_f1), "test_accuracy": float(t.test_accuracy)}
        t = g[g.variant == "after"].sort_values("val_macro_f1", ascending=False).iloc[0]
        pu = p_un if t.run_id == rid else load(t.run_id, dev)[0].predict_proba(un_paths)
        mu = metrics(y_un, pu)
        rows.append({"run": f"{mo} | tuned (seed {int(t.seed)})" + (" ← แอป" if t.run_id == rid else ""),
                     **{k: mu[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}, "loss": ce(y_un, pu)})
        print(f"Unseen {mo} หลังจูน: acc {mu['accuracy']:.3f} F1 {mu['macro_f1']:.3f}")
    pd.concat([um, pd.DataFrame(rows)]).to_csv(R / "unseen_metrics.csv", index=False)
    (R / "others" / "confusion_4models.json").write_text(
        json.dumps({"classes": fl.CLASSES, "models": conf}, indent=1), encoding="utf-8")
    (R / "deploy_selection.json").write_text(json.dumps(
        {"run_id": rid, "model": name, "seed": int(h["seed"]), "note": SELECTION_NOTE,
         "val_macro_f1_mean_after": summ[summ.variant == "after"].groupby("model").val_macro_f1.mean().round(4).to_dict(),
         "test_macro_f1_mean_after": summ[summ.variant == "after"].groupby("model").test_macro_f1.mean().round(4).to_dict()},
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
