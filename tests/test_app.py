import shutil
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "app_src"))
PAGES = ["page_compare", "page_tuning", "page_predict", "page_unseen"]
SMOKE_RES, SMOKE_MOD = ROOT / "results" / "_smoke", ROOT / "models" / "_smoke"


def _run_page(page, app_root, monkeypatch):
    monkeypatch.setenv("FABRIC_APP_ROOT", str(app_root))

    def script(app_dir, project_dir, page_name):
        import sys
        for p in (app_dir, project_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        import app
        importlib.reload(app)
        getattr(app, page_name)()

    at = AppTest.from_function(script, args=(str(ROOT / "app_src"), str(ROOT), page), default_timeout=120)
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_pages_without_results_show_hint(page, tmp_path, monkeypatch):
    at = _run_page(page, tmp_path, monkeypatch)
    assert not at.exception
    assert any("รัน notebook ส่วนที่" in i.value for i in at.info)


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    if not (SMOKE_RES / "test_before_after.csv").exists() or not (SMOKE_MOD / "model_meta.json").exists():
        pytest.skip("ต้องรัน notebook แบบ SMOKE ก่อน")
    root = tmp_path_factory.mktemp("approot")
    shutil.copytree(SMOKE_RES, root / "results")
    shutil.copytree(SMOKE_MOD, root / "models")
    if (ROOT / "unseen_data" / "sources.csv").exists():  # หน้า Unseen อ่าน sources.csv (รูปไม่จำเป็นต่อการทดสอบ)
        (root / "unseen_data").mkdir()
        shutil.copy(ROOT / "unseen_data" / "sources.csv", root / "unseen_data" / "sources.csv")
    return root


@pytest.mark.parametrize("page", PAGES)
def test_pages_with_smoke_results(page, smoke_root, monkeypatch):
    at = _run_page(page, smoke_root, monkeypatch)
    assert not at.exception, at.exception
    assert not any("รัน notebook ส่วนที่" in i.value for i in at.info)
    if page != "page_predict":
        assert len(at.get("plotly_chart")) >= 1


class _DummyPredictor:
    classes = ["Cotton", "Polyester", "Silk", "Viscose"]

    def predict_proba(self, images):
        return np.array([[0.1, 0.7, 0.1, 0.1]] * len(images))


def test_predict_bytes_bad_image():
    import app

    res = app.predict_bytes(_DummyPredictor(), b"not an image")
    assert isinstance(res, str) and "ไม่ใช่รูปภาพ" in res


def test_predict_bytes_good_image():
    import io

    from PIL import Image

    import app

    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (10, 200, 10, 128)).save(buf, "PNG")
    cls, p, img = app.predict_bytes(_DummyPredictor(), buf.getvalue())
    assert cls == "Polyester" and p.shape == (4,) and img.mode == "RGB"


def test_predict_bytes_too_large_message(monkeypatch):
    """minor #8: รูปใหญ่เกินต้องบอกว่าใหญ่เกิน ไม่ใช่ไฟล์เสีย"""
    import io

    from PIL import Image

    import app
    import fabric_lib as fl

    monkeypatch.setattr(fl, "MAX_PIXELS", 1000)
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "PNG")
    res = app.predict_bytes(_DummyPredictor(), buf.getvalue())
    assert isinstance(res, str) and "ใหญ่เกิน" in res


def test_unseen_page_with_results(tmp_path, monkeypatch):
    """R6: หน้า Unseen แสดงผลเมื่อมีไฟล์ครบ (ไม่ต้องรัน notebook ทั้งหมด)"""
    import pandas as pd
    from PIL import Image

    res, un = tmp_path / "results", tmp_path / "unseen_data" / "Cotton"
    res.mkdir()
    un.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(un / "cotton_000.jpg")
    pd.DataFrame([{"file": "Cotton/cotton_000.jpg", "class": "Cotton", "pred": "Silk", "conf": 0.7,
                   "p_Cotton": 0.1, "p_Polyester": 0.1, "p_Silk": 0.7, "p_Viscose": 0.1}]).to_csv(res / "unseen_predictions.csv", index=False)
    pd.DataFrame([{"run": "ResNet50 | after (seed 42)", "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
                   "macro_f1": 0.0, "loss": 2.3}]).to_csv(res / "unseen_metrics.csv", index=False)
    (tmp_path / "unseen_data" / "sources.csv").write_text(
        "file,class,source_url,textilenet_file,md5,description,label_source,screened_by\n"
        "Cotton/cotton_000.jpg,Cotton,http://x,a.jpg,m,desc,textilenet,claude\n", encoding="utf-8")
    at = _run_page("page_unseen", tmp_path, monkeypatch)
    assert not at.exception, at.exception
    assert not any("รัน notebook ส่วนที่" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) >= 1


def test_tuning_page_all_models(tmp_path, monkeypatch):
    """หน้า ก่อน vs หลังจูน แสดงครบ 4 โมเดลจาก results/others/ (ไม่ต้องมีไฟล์ของ notebook หลัก)"""
    import json

    import pandas as pd

    oth = ROOT / "results" / "others"
    if not (oth / "summary_4models.csv").exists() or not (oth / "confusion_4models.json").exists():
        pytest.skip("ต้องรัน tools/plot_before_after_4models.py และ tools/export_deploy_model.py ก่อน")
    res = tmp_path / "results"
    (res / "others").mkdir(parents=True)
    (res / "histories").mkdir()
    (tmp_path / "models").mkdir()
    for f in ("summary_4models.csv", "confusion_4models.json"):
        shutil.copy(oth / f, res / "others" / f)
    for f in oth.glob("optuna_trials_*.csv"):
        shutil.copy(f, res / "others" / f.name)
    for rid in pd.read_csv(oth / "summary_4models.csv").run_id:
        shutil.copy(ROOT / "results" / "histories" / f"{rid}.json", res / "histories")
    shutil.copy(ROOT / "models" / "model_meta.json", tmp_path / "models")
    at = _run_page("page_tuning", tmp_path, monkeypatch)
    assert not at.exception, at.exception
    assert not any("รัน notebook ส่วนที่" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) >= 6  # Val, Test, 2 curves, 2 confusion
    app_model = json.loads((tmp_path / "models" / "model_meta.json").read_text(encoding="utf-8"))["model_name"]
    assert at.selectbox[0].value in app_model  # ค่าเริ่มต้น = โมเดลที่แอปใช้
    for mo in at.selectbox[0].options:  # ทุกโมเดลต้องแสดงได้
        at.selectbox[0].set_value(mo).run()
        assert not at.exception, (mo, at.exception)
