"""FabricSort: แอป Streamlit 4 หน้า (เปรียบเทียบ 4 โมเดล / ก่อน vs หลังจูน / ทำนายรูป / Unseen)

สร้างจาก fabric_classification.ipynb (ส่วนที่ 13) รันด้วย: streamlit run app.py
อ่านเฉพาะไฟล์ใน results/ และ models/ ที่ notebook สร้าง ไม่คำนวณผลใหม่
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(os.environ.get("FABRIC_APP_ROOT") or Path(__file__).resolve().parent)
RES, MOD = ROOT / "results", ROOT / "models"
CLASSES = ["Cotton", "Polyester", "Silk", "Viscose"]
VARIANT_TH = {"before": "ก่อนจูน", "after": "หลังจูน"}
ROUNDS = ["none", "v1", "v2", "texture"]
UNS = ROOT / "unseen_data"
TH_NAME = {"Cotton": "ผ้าฝ้าย", "Polyester": "โพลีเอสเตอร์", "Silk": "ผ้าไหม", "Viscose": "วิสคอส (เรยอน)"}
ADVICE = {
    "Cotton": "เส้นใยธรรมชาติ (เซลลูโลส) รีไซเคิลเชิงกลได้ดี: ตัด/ฉีกเป็นเส้นใยแล้วปั่นเป็นด้ายใหม่ "
              "หรือทำเป็นวัสดุบุ ฉนวน และผ้าเช็ด ควรแยกตามสีเพื่อลดการฟอกย้อมซ้ำ",
    "Polyester": "พลาสติกชนิด PET: รีไซเคิลเชิงกลโดยหลอมเป็นเม็ดแล้วรีดเป็นเส้นใยใหม่ "
                 "หรือรีไซเคิลเชิงเคมี (depolymerization) ย่อยกลับเป็นโมโนเมอร์เพื่อผลิตโพลีเอสเตอร์คุณภาพเท่าเดิม",
    "Silk": "เส้นใยโปรตีนมูลค่าสูง: เหมาะกับการนำกลับมาใช้ใหม่หรือ upcycling "
            "หรือรีไซเคิลเชิงกลเป็นเส้นใยสั้นสำหรับผ้าผสม ไม่ควรปนกับผ้าใยสังเคราะห์",
    "Viscose": "เซลลูโลสที่ผ่านกระบวนการทางเคมี (regenerated cellulose): เส้นใยสั้นและอ่อนลงเมื่อรีไซเคิลเชิงกล "
               "จึงเหมาะกับรีไซเคิลเชิงเคมี คือละลายเป็นเยื่อเซลลูโลสแล้วผลิตเส้นใยใหม่ (เช่น lyocell)",
}
METRIC_TH = {"val_accuracy": "Accuracy", "val_macro_precision": "Precision", "val_macro_recall": "Recall",
             "val_macro_f1": "Macro-F1", "val_min_recall": "Recall ต่ำสุด"}


# ---------- อ่านไฟล์ผล ----------
def need(files, section) -> bool:
    """ถ้าไฟล์ใดยังไม่มี แสดงว่าต้องรัน notebook ส่วนไหน แล้วคืน False"""
    missing = [Path(f).name for f in files if not Path(f).exists()]
    if missing:
        st.info(f"ยังไม่มีผล ({', '.join(missing)}): รัน notebook ส่วนที่ {section} ก่อน แล้วรีเฟรชหน้านี้")
        return False
    return True


@st.cache_data
def _read_csv(path, mtime):
    return pd.read_csv(path)


@st.cache_data
def _read_json(path, mtime):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), path.stat().st_mtime)


def js(path: Path) -> dict:
    return _read_json(str(path), path.stat().st_mtime)


def model_meta():
    p = MOD / "model_meta.json"
    return js(p) if p.exists() else None


def sidebar():
    with st.sidebar:
        st.header("ข้อมูลโมเดล")
        meta = model_meta()
        if meta is None:
            st.caption("ยังไม่มีโมเดลที่ export (รัน notebook ส่วนที่ 12)")
        else:
            st.markdown(f"**โมเดล:** {meta['model_name']}")
            tm = meta["test_metrics"]
            c1, c2 = st.columns(2)
            c1.metric("Accuracy (Test)", f"{tm['accuracy']:.1%}")
            c2.metric("Macro-F1 (Test)", f"{tm['macro_f1']:.3f}")
            st.caption(f"ทดสอบกับ {meta['n_images']['test']:,} รูป · export เมื่อ {meta['exported_at']}")
        st.subheader("ข้อจำกัด")
        st.markdown(
            "- จำแนกได้เพียง 4 ชนิด: Cotton, Polyester, Silk, Viscose\n"
            "- ไม่รองรับผ้าผสม (blend) หรือผ้าชนิดอื่น ระบบจะตอบเป็น 1 ใน 4 ชนิดเสมอ\n"
            "- 1 รูปควรมีผ้าชิ้นเดียว ถ่ายใกล้และเต็มเฟรม\n"
            "- เทรนจากชุดข้อมูลสาธารณะ ความแม่นยำกับรูปจากโรงงานจริงอาจต่ำกว่านี้\n"
            "- ความมั่นใจต่ำ ควรให้คนตรวจซ้ำ\n"
            "- รูปประมวลผลในเครื่องนี้เท่านั้น ไม่ถูกบันทึกหรือส่งออกไปที่อื่น"
        )


# ---------- หน้า 1: เปรียบเทียบ 4 โมเดล ----------
def page_compare():
    st.title("เปรียบเทียบ 4 โมเดล × 4 รอบ")
    st.caption("none = ไม่ทำ augmentation, v1 = แบบเบา, v2 = แบบหนัก (ตามสไลด์), texture = แบบรักษาพื้นผิวผ้า · "
               "สูตร default เดียวกัน (seed 42) · "
               "เลือก Top 1 จากชุด Validation")
    if not need([RES / "screening.csv", RES / "selection.json"], "5–6"):
        return
    df, sel = csv(RES / "screening.csv"), js(RES / "selection.json")
    df["run"] = df.model + " | " + df["round"]
    top = df[df.candidate == sel["top1_candidate"]].iloc[0]
    st.success(f"🏆 Top 1: **{sel['top1_name']} รอบ {sel['top1_round']}**")
    meta = model_meta()
    if meta and meta.get("arch") != sel.get("top1_arch"):
        st.caption(f"ขั้นนี้คือรอบคัดเลือกด้วยสูตรเริ่มต้น · หลังจูนครบ 4 โมเดล แอปเปลี่ยนไปใช้ **{meta['model_name']}** "
                   "(ดูหน้า ก่อน vs หลังจูน)")
    cols = st.columns(4)
    cols[0].metric("Val Macro-F1", f"{top.val_macro_f1:.3f}")
    cols[1].metric("Val Accuracy", f"{top.val_accuracy:.1%}")
    cols[2].metric("Inference / รูป", f"{top.infer_ms_per_img:.0f} ms")
    cols[3].metric("ขนาดไฟล์", f"{top.size_mb:.0f} MB")

    fig = px.bar(df, x="model", y="val_macro_f1", color="round", barmode="group", text_auto=".3f",
                 category_orders={"round": ROUNDS}, title="Val Macro-F1: โมเดล × รอบ augmentation")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    long = df.melt(id_vars="run", value_vars=list(METRIC_TH), var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(METRIC_TH)
    fig = px.bar(long, x="metric", y="value", color="run", barmode="group", title="ตัวชี้วัดทั้งหมดบน Validation (16 run)")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ตารางผลทั้ง 16 run")
    show = df[["model", "round", "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
               "val_min_recall", "epochs_trained", "train_time_s", "fit"]]
    if (RES / "screening_test.csv").exists():
        show = show.merge(csv(RES / "screening_test.csv")[["model", "round", "test_accuracy", "test_macro_f1"]],
                          on=["model", "round"], how="left")
        st.caption("คอลัมน์ test_* คือผลบนชุด Test ที่รายงานหลังเลือกโมเดลแล้ว ไม่ได้ใช้ในการเลือก")
    st.dataframe(show.round(3), hide_index=True, use_container_width=True)

    st.subheader("ขนาดโมเดล")
    st.dataframe(df.drop_duplicates("model")[["model", "size_mb", "params_M", "imagenet_top1", "imagenet_top5",
                                              "infer_ms_per_img"]]
                 .rename(columns={"size_mb": "ขนาด (MB)", "params_M": "พารามิเตอร์ (M)", "imagenet_top1": "ImageNet Top-1 (%)",
                                  "imagenet_top5": "ImageNet Top-5 (%)", "infer_ms_per_img": "inference (ms/รูป)"}).round(2),
                 hide_index=True)

    st.subheader("รายละเอียดราย run")
    run = st.selectbox("เลือก run", df.run.tolist(), index=int(np.flatnonzero(df.candidate == sel["top1_candidate"])[0]))
    row = df[df.run == run].iloc[0]
    st.markdown(f"**วินิจฉัย:** {row.fit}")
    gpath = RES / "generalization.csv"
    if gpath.exists():
        g = csv(gpath)
        g = g[g.run_id == row.run_id].drop_duplicates("split")
        if len(g):
            gl = g.melt(id_vars="split", value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"],
                        var_name="metric", value_name="value")
            st.plotly_chart(px.bar(gl, x="metric", y="value", color="split", barmode="group", text_auto=".2f",
                                   title="Generalization: Train vs Val vs Test vs Unseen"), use_container_width=True)
    hpath = RES / "histories" / f"{row.run_id}.json"
    if need([hpath], "5"):
        h = pd.DataFrame(js(hpath)["history"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(h, x="epoch", y=[c for c in ("train_loss", "val_loss") if c in h], markers=True,
                                title="Learning curve: loss"), use_container_width=True)
        c2.plotly_chart(px.line(h, x="epoch", y=[c for c in ("train_accuracy", "val_accuracy") if c in h], markers=True,
                                title="Learning curve: accuracy"), use_container_width=True)
    rec = df[df.run == run].melt(value_vars=[f"recall_{c}" for c in CLASSES], var_name="class", value_name="recall")
    rec["class"] = rec["class"].str.replace("recall_", "")
    fig = px.bar(rec, x="class", y="recall", text_auto=".2f", title=f"Recall รายคลาส (Val): {run}")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**เกณฑ์เลือก:** Val Macro-F1 สูงสุด ถ้าต่างกันไม่เกิน 0.01 เลือกตัวที่ inference เร็วกว่า · "
                f"กลุ่มที่ต่างไม่เกิน 0.01: {', '.join(sel['candidates_within_0.01'])}")


# ---------- หน้า 2: ก่อน vs หลังจูน ----------
OTH = RES / "others"
OPTUNA_FILES = {"resnet50": RES / "optuna_trials.csv"}  # ตัวอื่นอยู่ที่ others/optuna_trials_<arch>.csv


def cm_figure(matrix, classes, title, scale="Blues"):
    m = np.array(matrix)
    norm = m / np.clip(m.sum(1, keepdims=True), 1, None)
    fig = px.imshow(norm, x=classes, y=classes, zmin=0, zmax=1, color_continuous_scale=scale,
                    labels={"x": "ทาย", "y": "จริง", "color": "สัดส่วน"}, title=title)
    fig.update_traces(text=m, texttemplate="%{text}<br>%{z:.0%}")
    return fig


def optuna_figure(tr, title):
    done = tr[tr.state == "COMPLETE"].sort_values("number")
    fig = go.Figure()
    fig.add_scatter(x=done.number, y=done.value, mode="markers", name="trial")
    fig.add_scatter(x=done.number, y=done.value.cummax(), mode="lines", name="ดีที่สุดถึงตอนนั้น")
    pruned = tr[tr.state == "PRUNED"]
    if len(pruned) and len(done):
        fig.add_scatter(x=pruned.number, y=[done.value.min()] * len(pruned), mode="markers", name="ถูกตัด (pruned)",
                        marker_symbol="x", marker_color="gray")
    fig.update_layout(title=title, xaxis_title="trial")
    return fig


def tuning_all_models():
    """ก่อน/หลังจูนครบ 4 โมเดล (results/others/ จาก tune_others.ipynb + tools/plot_before_after_4models.py)"""
    summ, conf = csv(OTH / "summary_4models.csv"), js(OTH / "confusion_4models.json")
    meta = model_meta() or {}
    st.caption("ก่อนจูน = สูตรเริ่มต้นของรอบ augmentation ที่ดีที่สุดของโมเดลนั้น · หลังจูน = hyperparameter ที่ดีที่สุดจาก "
               "Optuna 15 trial · ทุกช่วงเทรน 3 seed (42, 7, 2024) · Test 328 รูป")
    summ["ชุด"] = summ.variant.map(VARIANT_TH)
    agg = summ.groupby(["model", "ชุด"], sort=False).agg(
        val_mean=("val_macro_f1", "mean"), val_std=("val_macro_f1", "std"),
        test_mean=("test_macro_f1", "mean"), test_std=("test_macro_f1", "std"),
        acc_mean=("test_accuracy", "mean")).reset_index()
    c1, c2 = st.columns(2)
    for c, (k, label) in zip((c1, c2), (("val", "Val Macro-F1"), ("test", "Test Macro-F1"))):
        fig = px.bar(agg, x="model", y=f"{k}_mean", color="ชุด", barmode="group", error_y=f"{k}_std",
                     text_auto=".3f", title=f"{label} (mean ± SD, 3 seed)",
                     color_discrete_map={"ก่อนจูน": "#4d7be0", "หลังจูน": "#ec6a3c"})
        fig.update_yaxes(range=[0.5, 1], title=label)
        c.plotly_chart(fig, use_container_width=True)
    tab = agg.pivot(index="model", columns="ชุด", values=["val_mean", "test_mean", "acc_mean"]).round(3)
    tab.columns = [f"{a.replace('_mean', '')} {b}" for a, b in tab.columns]
    st.dataframe(tab, use_container_width=True)
    if meta.get("selection_note"):
        st.info(f"**โมเดลที่ใช้ในแอป:** {meta['model_name']} seed {meta['seed']}  \n{meta['selection_note']}")

    st.subheader("รายละเอียดรายโมเดล")
    models = list(dict.fromkeys(summ.model))
    app_name = next((m for m in models if meta.get("model_name", "").startswith(m)), models[0])
    mo = st.selectbox("เลือกโมเดล", models, index=models.index(app_name))
    d = summ[summ.model == mo]
    rows = []
    for t in d.to_dict("records"):
        hp = RES / "histories" / f"{t['run_id']}.json"
        if hp.exists():
            h = pd.DataFrame(js(hp)["history"])
            rows.append(h.assign(run=f"{t['ชุด']} s{t['seed']}", ชุด=t["ชุด"]))
    if rows:
        hh = pd.concat(rows)
        c1, c2 = st.columns(2)
        for c, (col, label) in zip((c1, c2), (("val_macro_f1", "Val Macro-F1"), ("val_loss", "Val loss"))):
            fig = px.line(hh, x="epoch", y=col, color="run", line_dash="ชุด", title=f"{mo}: {label}")
            c.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    for c, v, scale in ((c1, "before", "Blues"), (c2, "after", "Oranges")):
        r = conf["models"][mo][v]
        c.plotly_chart(cm_figure(r["matrix"], conf["classes"], f"{VARIANT_TH[v]} (seed {r['seed']}): Test acc "
                                 f"{r['test_accuracy']:.3f} · Macro-F1 {r['test_macro_f1']:.3f}", scale),
                       use_container_width=True)
    params = {}
    for v in ("before", "after"):
        hp = RES / "histories" / f"{conf['models'][mo][v]['run_id']}.json"
        if hp.exists():
            params[VARIANT_TH[v]] = js(hp)["params"]
            arch = js(hp)["arch"]
    if params:
        st.markdown("**Hyperparameter**")
        st.dataframe(pd.DataFrame(params).astype(str), use_container_width=True)
        op = OPTUNA_FILES.get(arch, OTH / f"optuna_trials_{arch}.csv")
        if op.exists():
            st.plotly_chart(optuna_figure(csv(op), f"{mo}: Optuna Val Macro-F1 ราย trial"), use_container_width=True)
        elif arch == "convnext_tiny":
            st.caption("ConvNeXt-Tiny จูนในรอบรันก่อนหน้า (ยังไม่มีรอบ texture) ไม่มีไฟล์ประวัติ trial ในรอบนี้")


def page_tuning():
    st.title("ก่อน vs หลังจูน")
    has_all = (OTH / "summary_4models.csv").exists() and (OTH / "confusion_4models.json").exists()
    if has_all:
        tuning_all_models()
    files = [RES / f for f in ("test_before_after.csv", "confusion_before.json", "confusion_after.json",
                               "per_class_test.csv", "per_source_test.csv", "significance.csv",
                               "optuna_trials.csv", "optuna_importance.csv", "runs.csv", "selection.json")]
    if has_all:
        if all(f.exists() for f in files):
            with st.expander("รายละเอียด Top 1 จาก notebook หลัก (ResNet50): ผลรายคลาส แหล่งข้อมูล สถิติ Optuna"):
                tuning_top1(files)
        return
    if not need(files, "7–10"):
        return
    tuning_top1(files)


def tuning_top1(files):
    tba, sel = csv(RES / "test_before_after.csv"), js(RES / "selection.json")
    st.caption(f"โมเดล: {sel['top1_name']} รอบ {sel.get('top1_round', '-')} · ก่อนจูน = สูตรของรอบที่ชนะ · "
               "หลังจูน = hyperparameter ที่ดีที่สุดจาก Optuna · 3 seed")
    fb = sel.get("fallback", {})
    if fb.get("needed"):
        st.warning(f"Test Macro-F1 หลังจูน {fb['after_test_macro_f1_mean']:.3f} ต่ำกว่าเป้า {fb['target']}: "
                   "ต้องทำขั้นสำรอง (ConvNeXt V2-Tiny)")

    tba["ชุด"] = tba.variant.map(VARIANT_TH)
    fig = go.Figure()
    for col, label in (("val_macro_f1", "Val Macro-F1"), ("test_macro_f1", "Test Macro-F1"), ("test_accuracy", "Test Accuracy")):
        g = tba.groupby("ชุด", sort=False)[col]
        fig.add_bar(name=label, x=list(g.mean().index), y=g.mean().values, error_y=dict(type="data", array=g.std(ddof=0).values),
                    text=[f"{m:.3f}" for m in g.mean().values], offsetgroup=label)
    fig.update_layout(barmode="group", title="mean ± std (3 seed)", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.strip(tba, x="ชุด", y="test_macro_f1", color="seed", title="Test Macro-F1 ราย seed"),
                    use_container_width=True)

    st.subheader("Confusion matrix (Test, seed ที่ Val ดีที่สุด)")
    cols = st.columns(2)
    for c, v in zip(cols, ("before", "after")):
        cm = js(RES / f"confusion_{v}.json")
        m = np.array(cm["matrix"])
        norm = m / np.clip(m.sum(1, keepdims=True), 1, None)
        fig = px.imshow(norm, x=cm["classes"], y=cm["classes"], zmin=0, zmax=1, color_continuous_scale="Blues",
                        labels={"x": "ทาย", "y": "จริง", "color": "สัดส่วน"}, title=VARIANT_TH[v])
        fig.update_traces(text=m, texttemplate="%{text}<br>%{z:.0%}")
        c.plotly_chart(fig, use_container_width=True)

    pc = csv(RES / "per_class_test.csv")
    wide = pc.pivot(index="class", columns="variant", values="f1").reindex(CLASSES)
    wide["ส่วนต่าง"] = wide["after"] - wide["before"]
    fig = px.bar(wide.reset_index(), x="class", y="ส่วนต่าง", text_auto="+.3f", title="F1 รายคลาส: หลัง − ก่อน (Test)",
                 color="ส่วนต่าง", color_continuous_scale="RdBu", range_color=[-0.2, 0.2])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(wide.rename(columns=VARIANT_TH).round(3))

    st.subheader("แยกตามแหล่งข้อมูล (Test)")
    ps = csv(RES / "per_source_test.csv")
    ps["variant"] = ps.variant.map(VARIANT_TH)
    st.dataframe(ps.round(3), hide_index=True)

    st.subheader("ความเชื่อมั่นทางสถิติ")
    sig = csv(RES / "significance.csv")
    sig["variant"] = sig.variant.map(VARIANT_TH)
    st.dataframe(sig[["variant", "seed", "test_macro_f1", "ci95_low", "ci95_high"]].round(3), hide_index=True)
    p = float(sig.mcnemar_p.iloc[0])
    st.markdown(f"**McNemar test** (หลังถูก/ก่อนผิด = {int(sig.mcnemar_after_right_before_wrong.iloc[0])}, "
                f"ก่อนถูก/หลังผิด = {int(sig.mcnemar_before_right_after_wrong.iloc[0])}): p = {p:.3f} → "
                + ("ต่างกันอย่างมีนัยสำคัญ (p < 0.05)" if p < 0.05 else "ยังไม่ต่างกันอย่างมีนัยสำคัญ (p ≥ 0.05)"))

    st.subheader("Optuna")
    tr = csv(RES / "optuna_trials.csv")
    done = tr[tr.state == "COMPLETE"].sort_values("number")
    fig = go.Figure()
    fig.add_scatter(x=done.number, y=done.value, mode="markers", name="trial")
    fig.add_scatter(x=done.number, y=done.value.cummax(), mode="lines", name="ดีที่สุดถึงตอนนั้น")
    pruned = tr[tr.state == "PRUNED"]
    if len(pruned):
        fig.add_scatter(x=pruned.number, y=[done.value.min()] * len(pruned), mode="markers", name="ถูกตัด (pruned)",
                        marker_symbol="x", marker_color="gray")
    fig.update_layout(title="Val Macro-F1 ราย trial", xaxis_title="trial")
    st.plotly_chart(fig, use_container_width=True)
    imp = csv(RES / "optuna_importance.csv")
    if len(imp):
        st.plotly_chart(px.bar(imp.sort_values("importance"), x="importance", y="param", orientation="h",
                               title="ความสำคัญของ hyperparameter"), use_container_width=True)
    runs = csv(RES / "runs.csv")
    pcols = [c for c in runs.columns if c.startswith("p_")]
    rows = {VARIANT_TH[s]: runs[runs.stage == s][pcols].iloc[0] for s in ("before", "after") if (runs.stage == s).any()}
    if rows:
        st.dataframe(pd.DataFrame(rows).rename(index=lambda c: c[2:]).astype(str), use_container_width=True)


# ---------- หน้า 3: ทำนายรูป ----------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def get_predictor(meta_mtime):
    import fabric_lib as fl
    return fl.load_predictor(MOD)


def predict_bytes(predictor, data):
    """bytes ของรูป -> (คลาส, probability 4 ค่า, PIL RGB) หรือข้อความ error (str) ถ้าใช้รูปนี้ไม่ได้"""
    import fabric_lib as fl
    try:
        img = fl.load_pil(data)
        img.load()
    except fl.ImageTooLargeError as e:
        return f"รูปใหญ่เกินไป ({e}) กรุณาย่อรูปหรือถ่ายด้วยความละเอียดต่ำลง"
    except Exception:
        return "ไม่ใช่รูปภาพที่รองรับหรือไฟล์เสีย กรุณาใช้ไฟล์ jpg, png หรือ webp"
    try:
        p = np.asarray(predictor.predict_proba([img])[0], dtype=float)
    except Exception:
        return "เกิดข้อผิดพลาดระหว่างจำแนกรูป กรุณาลองใหม่อีกครั้งหรือใช้รูปอื่น"
    return predictor.classes[int(np.argmax(p))], p, img


def page_predict():
    st.title("🧵 ทำนายชนิดผ้าจากรูป")
    if not need([MOD / "model_meta.json", MOD / "best_model.pth"], "12"):
        return
    try:
        predictor = get_predictor((MOD / "model_meta.json").stat().st_mtime)
    except Exception:
        st.error("โหลดโมเดลไม่สำเร็จ กรุณารัน notebook ส่วนที่ 12 (Export) ใหม่อีกครั้ง")
        return
    threshold = float(predictor.meta.get("conf_threshold", 0.6))
    with st.expander("📸 คำแนะนำการถ่ายรูป"):
        st.markdown("1. ถ่ายใกล้ ๆ ให้ผ้าเต็มเฟรม ไม่มีพื้นหลังหรือวัตถุอื่น\n2. ใช้แสงสม่ำเสมอ หลีกเลี่ยงเงาและแสงสะท้อนจ้า\n"
                    "3. ถือกล้องให้นิ่ง ให้เห็นลายเส้นใยชัดเจน\n4. 1 รูปต่อผ้า 1 ชิ้น")
    tab_up, tab_cam = st.tabs(["📁 อัปโหลดรูป", "📷 ถ่ายรูป"])
    with tab_up:
        files = st.file_uploader("เลือกรูปผ้า (jpg, png, webp, bmp)", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                 accept_multiple_files=True)
    with tab_cam:  # เปิดกล้องเมื่อผู้ใช้เลือกเท่านั้น (ไม่ขอสิทธิ์กล้องตั้งแต่เปิดหน้า)
        shot = st.camera_input("ถ่ายรูปผ้า") if st.toggle("เปิดกล้อง") else None
    inputs = list(files or []) + ([shot] if shot is not None else [])
    if not inputs:
        st.info("อัปโหลดหรือถ่ายรูปผ้าเพื่อเริ่มจำแนก")
        return
    for f in inputs:
        res = predict_bytes(predictor, f.getvalue())
        if isinstance(res, str):
            st.error(f"ไฟล์ “{f.name}”: {res}")
            continue
        cls, p, img = res
        conf = float(p.max())
        c_img, c_res = st.columns([1, 1.3])
        c_img.image(img, caption=f.name)
        with c_res:
            st.subheader(f"{cls} · {TH_NAME.get(cls, cls)}")
            st.metric("ความมั่นใจ", f"{conf:.1%}")
            if conf < threshold:
                st.warning(f"⚠️ ความมั่นใจต่ำกว่า {threshold:.0%} ควรให้คนตรวจซ้ำ")
            fig = px.bar(x=predictor.classes, y=p * 100, text=[f"{v:.1%}" for v in p],
                         labels={"x": "ชนิดผ้า", "y": "ความน่าจะเป็น (%)"})
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"♻️ **การรีไซเคิล:** {ADVICE.get(cls, '-')}")


# ---------- หน้า 4: Unseen data ----------
def page_unseen():
    st.title("🌐 Unseen data (รูปจากอินเทอร์เน็ต)")
    st.caption("รูปสินค้าผ้าจาก TextileNet (Zhong et al., 2023) ต่างแหล่งกับชุดเทรน · label จาก TextileNet · "
               "คัดกรองรูปโดย AI · ใช้วัดผลอย่างเดียว ไม่ได้ใช้เทรนหรือเลือกโมเดล")
    if not need([UNS / "sources.csv", RES / "unseen_predictions.csv", RES / "unseen_metrics.csv"], "10b–10c"):
        return
    up, um = csv(RES / "unseen_predictions.csv"), csv(RES / "unseen_metrics.csv")
    ok = up["class"] == up["pred"]
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนรูป", f"{len(up)}")
    c2.metric("Accuracy (โมเดลที่ใช้ในแอป)", f"{ok.mean():.1%}")
    f1 = []
    for c in CLASSES:
        tp = ((up["class"] == c) & (up.pred == c)).sum()
        prec = tp / max((up.pred == c).sum(), 1)
        rec = tp / max((up["class"] == c).sum(), 1)
        f1.append({"class": c, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                   "n": int((up["class"] == c).sum())})
    f1 = pd.DataFrame(f1)
    c3.metric("Macro-F1", f"{f1.f1.mean():.3f}")

    fig = px.bar(um, x="run", y="macro_f1", text_auto=".3f", title="Macro-F1 บน Unseen ของทุก run")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.bar(f1.melt(id_vars="class", value_vars=["precision", "recall", "f1"]), x="class", y="value",
                           color="variable", barmode="group", text_auto=".2f", title="รายคลาส (โมเดลที่ใช้ในแอป)"),
                    use_container_width=True)
    cm = pd.crosstab(pd.Categorical(up["class"], CLASSES), pd.Categorical(up.pred, CLASSES), dropna=False)
    norm = cm.div(cm.sum(1).clip(lower=1), axis=0)
    fig = px.imshow(norm.values, x=CLASSES, y=CLASSES, zmin=0, zmax=1, color_continuous_scale="Blues",
                    labels={"x": "ทาย", "y": "จริง", "color": "สัดส่วน"}, title="Confusion matrix (Unseen)")
    fig.update_traces(text=cm.values, texttemplate="%{text}<br>%{z:.0%}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ดูรูปทีละรูป")
    src = csv(UNS / "sources.csv")[["file", "description"]]
    view = up.merge(src, on="file", how="left")
    f_cls = st.selectbox("คลาสจริง", ["ทั้งหมด"] + CLASSES)
    only_wrong = st.toggle("เฉพาะที่ทายผิด", value=True)
    if f_cls != "ทั้งหมด":
        view = view[view["class"] == f_cls]
    if only_wrong:
        view = view[view["class"] != view.pred]
    st.caption(f"{len(view)} รูป")
    cols = st.columns(4)
    for k, r in enumerate(view.head(24).itertuples()):
        with cols[k % 4]:
            path = UNS / r.file
            if path.exists():
                st.image(str(path), use_container_width=True)
            st.caption(f"จริง **{r._2}** → ทาย **{r.pred}** ({r.conf:.0%})  \n{r.description if isinstance(r.description, str) else ''}")


def main():
    st.set_page_config(page_title="FabricSort", page_icon="🧵", layout="wide")
    sidebar()
    nav = st.navigation([st.Page(page_compare, title="เปรียบเทียบ 4 โมเดล", icon="📊", default=True),
                         st.Page(page_tuning, title="ก่อน vs หลังจูน", icon="🎛️"),
                         st.Page(page_predict, title="ทำนายรูป", icon="🧵"),
                         st.Page(page_unseen, title="Unseen data", icon="🌐")])
    nav.run()


if __name__ == "__main__":
    main()
