"""กราฟรวมก่อน/หลังจูนของทั้ง 4 โมเดล (learning curve + Val/Test F1 + confusion matrix บน Test)

รันหลัง tune_others.ipynb เสร็จ: .venv\\Scripts\\python.exe -I tools\\plot_before_after_4models.py
ผล: results/before_after_4models.png, results/confusion_before_after_4models.png, results/others/summary_4models.csv
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

import fabric_lib as fl  # noqa: E402

R = ROOT / "results"
SEEDS = (42, 7, 2024)
CACHE = R / "others" / "test_pred_cache.json"


def run_table() -> pd.DataFrame:
    rows = []
    tba = pd.read_csv(R / "test_before_after.csv")  # ResNet50 (Top 1, notebook หลัก)
    for t in tba.itertuples():
        rows.append(("ResNet50", "texture", t.variant, t.seed, t.run_id))
    conv = {"before": ["fe9354840d00", "6e09f4f353d6", "c6bb3ea6e2f1"],  # รอบรันก่อนหน้า (ConvNeXt เป็น Top 1)
            "after": ["f5e24fd4379c", "18493058ddc2", "b5c35bfea0d1"]}
    for v, ids in conv.items():
        rows += [("ConvNeXt-Tiny", "none", v, s, r) for s, r in zip(SEEDS, ids)]
    oth = pd.read_csv(R / "others" / "before_after_others.csv")
    rows += [(t.model, t.round, t.variant, t.seed, t.run_id) for t in oth.itertuples()]
    return pd.DataFrame(rows, columns=["model", "round", "variant", "seed", "run_id"])


def test_set():
    sp = pd.read_csv(R / "split.csv")  # Val/Test 1 รูปต่อชิ้นผ้า (เหมือน notebook)
    ev = sp[sp.split != "train"]
    one = ev.sort_values("path").groupby("sample_id").sample(1, random_state=42)
    te = one[one.split == "test"].sort_values("path").reset_index(drop=True)
    assert len(te) == 328, len(te)
    return [ROOT / "dataset" / p for p in te.path], te["class"].map({c: i for i, c in enumerate(fl.CLASSES)}).to_numpy()


def predict(df, paths):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for rid in df.run_id:
        if rid in cache:
            continue
        h = json.loads((R / "histories" / f"{rid}.json").read_text(encoding="utf-8"))
        m = fl.build_model(h["arch"], dropout=h["params"]["dropout"], pretrained=False)
        m.load_state_dict(torch.load(ROOT / "models" / "runs" / f"{rid}.pth", map_location="cpu", weights_only=True))
        cache[rid] = fl.TorchPredictor(m, dev, batch_size=32).predict_proba(paths).argmax(1).tolist()
        CACHE.write_text(json.dumps(cache))
        print("predicted", rid, flush=True)
    return cache


def main():
    df = run_table()
    paths, y = test_set()
    pred = predict(df, paths)
    hist = {r: json.loads((R / "histories" / f"{r}.json").read_text(encoding="utf-8"))["history"] for r in df.run_id}
    df["val_macro_f1"] = [max(x["val_macro_f1"] for x in hist[r]) for r in df.run_id]
    df["test_accuracy"] = [accuracy_score(y, pred[r]) for r in df.run_id]
    df["test_macro_f1"] = [f1_score(y, pred[r], average="macro") for r in df.run_id]
    df.to_csv(R / "others" / "summary_4models.csv", index=False)
    models = ["ResNet50", "ConvNeXt-Tiny", "DenseNet121", "EfficientNet-B0"]
    agg = df.groupby(["model", "variant"])[["val_macro_f1", "test_macro_f1", "test_accuracy"]].agg(["mean", "std"])
    print(agg.round(3).to_string())

    # ---- รูป 1: learning curves + แท่ง Val/Test ----
    col = {"before": ["#1c3f95", "#4d7be0", "#9db8f2"], "after": ["#b02a1a", "#ec6a3c", "#f5ab8c"]}
    panels = [("val_macro_f1", "Val Macro-F1"), ("val_loss", "Val loss"), ("train_loss", "Train loss")]
    fig, ax = plt.subplots(4, 4, figsize=(24, 19), gridspec_kw={"width_ratios": [1, 1, 1, .9]})
    for i, mo in enumerate(models):
        d = df[df.model == mo]
        aug = {v: json.loads((R / "histories" / f"{d[d.variant == v].run_id.iloc[0]}.json").read_text(encoding="utf-8"))["params"]["aug"]
               for v in ("before", "after")}
        rnd = aug["before"] if aug["before"] == aug["after"] else f"{aug['before']} -> {aug['after']}"
        for v in ("before", "after"):
            for c, t in zip(col[v], d[d.variant == v].itertuples()):
                h = hist[t.run_id]
                for j, (k, title) in enumerate(panels):
                    ax[i, j].plot([x["epoch"] for x in h], [x[k] for x in h], color=c, lw=1.7, label=f"{v} s{t.seed}")
        for j, (k, title) in enumerate(panels):
            ax[i, j].set_title(f"{mo} (aug {rnd}): {title}", fontsize=11)
            ax[i, j].grid(alpha=.3)
            ax[i, j].set_xlabel("epoch")
        ax[i, 0].legend(fontsize=8, ncol=2)
        x = np.arange(2)
        for k, (met, lab) in enumerate((("val_macro_f1", "Val"), ("test_macro_f1", "Test"))):
            mu = [agg.loc[(mo, v), (met, "mean")] for v in ("before", "after")]
            sd = [agg.loc[(mo, v), (met, "std")] for v in ("before", "after")]
            bars = ax[i, 3].bar(x + (k - .5) * .38, mu, .36, yerr=sd, capsize=5, label=lab,
                                color=["#4d7be0", "#ec6a3c"], alpha=1 if k == 0 else .55,
                                edgecolor="black", hatch="" if k == 0 else "//")
            for b, m_ in zip(bars, mu):
                ax[i, 3].text(b.get_x() + b.get_width() / 2, m_ + .015, f"{m_:.3f}", ha="center", fontsize=9)
        ax[i, 3].set_xticks(x, ["before", "after"])
        ax[i, 3].set_ylim(.6, .92)
        ax[i, 3].set_title(f"{mo}: Macro-F1 mean of 3 seeds (solid = Val, hatched = Test)", fontsize=10)
        ax[i, 3].grid(alpha=.3, axis="y")
    fig.suptitle("Before vs after Optuna fine-tuning, all 4 models (blue = before, red = after)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(R / "before_after_4models.png", dpi=85)
    plt.close(fig)

    # ---- รูป 2: confusion matrix (seed ที่ Val ดีที่สุดของแต่ละช่วง เหมือน notebook) ----
    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    for j, mo in enumerate(models):
        for i, v in enumerate(("before", "after")):
            t = df[(df.model == mo) & (df.variant == v)].sort_values("val_macro_f1", ascending=False).iloc[0]
            cm = confusion_matrix(y, pred[t.run_id], labels=range(4))
            pct = cm / cm.sum(1, keepdims=True)
            a = ax[i, j]
            a.imshow(pct, cmap="Blues" if v == "before" else "Oranges", vmin=0, vmax=1)
            for r in range(4):
                for c in range(4):
                    a.text(c, r, f"{cm[r, c]}\n{pct[r, c]:.0%}", ha="center", va="center", fontsize=10,
                           color="white" if pct[r, c] > .6 else "black")
            a.set_xticks(range(4), fl.CLASSES, rotation=30)
            a.set_yticks(range(4), fl.CLASSES)
            a.set_xlabel("predicted")
            a.set_ylabel("true")
            a.set_title(f"{mo} {v} tuning (seed {t.seed})\nTest acc {t.test_accuracy:.3f} | Macro-F1 {t.test_macro_f1:.3f}",
                        fontsize=11)
    fig.suptitle("Confusion matrix on Test (n=328): before (top) vs after (bottom) tuning", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=3)
    fig.savefig(R / "confusion_before_after_4models.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
