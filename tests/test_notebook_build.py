import json
import re
import subprocess
import sys

import nbformat
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import build_notebook as bn  # noqa: E402

SECTIONS = ["## 0. Config", "## 1. Setup", "## 2. Load + EDA", "## 3. Clean + Split", "## 4.", "## 5.", "## 6.",
            "## 7.", "## 8.", "## 9.", "## 10.", "## 10b.", "## 10c.", "## 11.", "## 12.", "## 13.", "## 14. เปิดแอป"]


@pytest.fixture(scope="module")
def nb(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py"), "--out", str(out)], check=True)
    return nbformat.read(out, as_version=4)


def test_builder_creates_sections(nb):
    heads = [l for c in nb.cells if c.cell_type == "markdown" for l in c.source.splitlines() if l.startswith("## ")]
    pos = []
    for s in SECTIONS:
        idx = [i for i, h in enumerate(heads) if h.startswith(s)]
        assert idx, f"missing section {s}"
        pos.append(idx[0])
    assert pos == sorted(pos)


def _config_src(nb):
    return next(c.source for c in nb.cells if c.cell_type == "code" and "# ===== Config" in c.source)


@pytest.mark.parametrize("smoke", [True, False])
def test_smoke_paths_isolated(nb, tmp_path, monkeypatch, smoke):
    monkeypatch.chdir(tmp_path)
    src = re.sub(r"^SMOKE = False", f"SMOKE = {smoke}", _config_src(nb), flags=re.M)
    g = {}
    exec(src, g)
    res, mod = g["RESULTS_DIR"].relative_to(tmp_path).as_posix(), g["MODELS_DIR"].relative_to(tmp_path).as_posix()
    assert g["SPLIT_DIR"].relative_to(tmp_path).as_posix() == "results"
    if smoke:
        assert (res, mod) == ("results/_smoke", "models/_smoke")
        assert g["N_TRIALS"] == 2 and g["MAX_EPOCHS"] == 2
    else:
        assert (res, mod) == ("results", "models")
        assert g["N_TRIALS"] == 15 and g["SEEDS"] == (42, 7, 2024)


def test_reused_cells_identical(nb):
    old = json.loads(bn.BACKUP_NB.read_text(encoding="utf-8"))["cells"]
    sources = [c.source for c in nb.cells]
    for i in bn.REUSED_CELLS:
        src = "".join(old[i]["source"])
        for a, b in bn.PATH_SUBS + bn.PLOT_SUBS:
            src = src.replace(a, b)
        assert src.strip() in sources, f"old cell {i} not reused verbatim"
    assert not any('RESULTS_DIR / "split.csv"' in s for s in sources)


def test_embedded_files_current(nb):
    sources = [c.source for c in nb.cells]
    lib = (ROOT / "fabric_lib.py").read_text(encoding="utf-8").strip()
    assert any(s.startswith("%%writefile fabric_lib.py") and lib in s for s in sources)
    assert any(s.startswith("%%writefile app.py") for s in sources)


def test_trial_run_id_differs_from_full_run(nb):
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "def run_id_of" in c.source)
    fn_src = src[src.index("def run_id_of"):src.index("def _fit")]
    import hashlib
    import json as _json
    g = {"hashlib": hashlib, "json": _json, "FINGERPRINT": "x"}
    exec(fn_src, g)
    p = {"a": 1}
    assert g["run_id_of"]("resnet50", p, 42, 2) != g["run_id_of"]("resnet50", p, 42, 2, trial=True)


def test_optuna_cell_frees_memory_and_reseeds(nb):
    """review #2/#3: sampler seed ขึ้นกับจำนวน trial ที่มีแล้ว และคืนหน่วยความจำหลังทุก trial"""
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "study.optimize" in c.source)
    assert "make_sampler(len(study.trials)" in src
    assert "gc_after_trial=True" in src
    assert "torch.cuda.empty_cache()" in src.split("def objective")[1]


def test_export_cell_uses_check_export(nb):
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "best_model.pth" in c.source and "shutil" in c.source)
    assert "check_export(" in src and "assert (p_app.argmax(1)" not in src


def test_cached_run_warns_on_code_version_change(nb):
    """minor #7: run ที่ cache ไว้จากโค้ดเวอร์ชันอื่นต้องมีคำเตือน (cache key ไม่รวมเวอร์ชันโค้ด)"""
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "def train_one" in c.source)
    assert "CODE_VERSION" in src and '"code_version": CODE_VERSION' in src
    assert "fail_stale_trials(study)" in next(c.source for c in nb.cells if c.cell_type == "code" and "study.optimize" in c.source)


def test_no_seaborn_anywhere(nb):
    """R5: notebook ใช้ matplotlib อย่างเดียว"""
    for c in nb.cells:
        if c.cell_type == "code":
            assert "seaborn" not in c.source and "sns." not in c.source, c.source[:80]


def _code(nb, needle):
    return next(c.source for c in nb.cells if c.cell_type == "code" and needle in c.source)


def test_rounds_generalization_unseen_cells(nb):
    """R5: รอบคัดเลือก 4 โมเดล × 3 รอบ, ส่วน generalization และ unseen"""
    cfg = _code(nb, "# ===== Config")
    assert 'ROUNDS = ["none", "v1", "v2", "texture"]' in cfg and "UNSEEN_DIR" in cfg
    scr = _code(nb, "SCREEN_RUNS")
    assert "for r in ROUNDS" in scr and "candidate" in scr and "imagenet_top1" in scr and "size_mb" in scr
    assert "generalization.csv" in _code(nb, "generalization.csv")
    un = _code(nb, "unseen_metrics.csv")
    assert "UNSEEN_DIR" in un and "unseen_predictions.csv" in un
    assert "screening_test.csv" in _code(nb, "screening_test.csv")
    assert "CODE_VERSION = 3" in _code(nb, "def train_one")
    assert '"val_loss"' in _code(nb, "def _fit") and '"train_accuracy"' in _code(nb, "def _fit")


def test_no_old_aug_names(nb):
    """R7 SMOKE: ค่า augmentation เดิม (base/strong) ต้องไม่เหลือใน notebook"""
    for c in nb.cells:
        if c.cell_type == "code" and not c.source.startswith("%%writefile"):
            assert 'aug="base"' not in c.source and '"strong"' not in c.source, c.source[:80]


def test_gen_filters_by_label_not_run_id(nb):
    """review C2: run_id ซ้ำได้ (ก่อนจูน seed 42 = run รอบคัดเลือก) จึงต้องกรองด้วยชื่อ run"""
    src = _code(nb, "generalization.csv")
    assert "generalization[generalization.run == label]" in src
    assert "generalization[generalization.run_id == rid]" not in src
    assert 'predict_split(rid, "train")' in _code(nb, "SCREEN_RUNS =")


def test_optuna_study_name_versioned(nb):
    """search space เปลี่ยน (เพิ่ม texture) -> ชื่อ study ต้องใหม่ ไม่งั้น Optuna โหลด study เก่าที่ค่าตัวเลือกต่างกัน"""
    assert 'study_name=f"top1_{TOP1}_{TOP1_ROUND}_ss2"' in _code(nb, "study.optimize")


def test_create_study_keywords_are_real_code(nb):
    """บั๊ก rev 3: คอมเมนต์ในบรรทัดเดียวกันทับอาร์กิวเมนต์ storage/direction -> study ในหน่วยความจำที่ minimize F1
    ตรวจจาก AST (โค้ดจริง) ไม่ใช่ข้อความ"""
    import ast

    src = _code(nb, "study.optimize")
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "create_study"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert {"study_name", "storage", "direction", "load_if_exists", "pruner"} <= set(kw)
    assert ast.literal_eval(kw["direction"]) == "maximize" and ast.literal_eval(kw["load_if_exists"]) is True


def test_gpu_cooldown_wired_in(nb):
    """มีจุดตรวจอุณหภูมิในการเทรนและวัดผล (ปิดไว้เป็นค่าเริ่มต้น ผู้ใช้สั่งหยุดเอง)"""
    cfg = _code(nb, "# ===== Config")
    # ผู้ใช้เปลี่ยนใจ: แจ้งเตือนอย่างเดียว ผู้ใช้สั่งหยุดเอง -> ปิดการพักอัตโนมัติเป็นค่าเริ่มต้น
    assert re.search(r"GPU_TEMP_PAUSE\s*=\s*None\s", cfg) and "GPU_TEMP_PAUSE is None" in _code(nb, "def cool_gpu")
    train = _code(nb, "def _fit")
    fit_src = train[train.index("def _fit"):train.index("def load_run")]
    pred_src = train[train.index("def predict_model"):train.index("def ce_loss")]
    assert "cool_gpu()" in fit_src and "cool_gpu()" in pred_src
