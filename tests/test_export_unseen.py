import csv
import sys

from PIL import Image

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import export_unseen as eu  # noqa: E402


def _setup(tmp_path):
    cand = tmp_path / "cand"
    rows = []
    for cls, n in (("Cotton", 4), ("Silk", 2)):
        (cand / cls).mkdir(parents=True)
        for i in range(n):
            f = f"{cls.lower()}_{i:03d}.jpg"
            Image.new("RGB", (200, 200), (i * 40, 0, 0)).save(cand / cls / f)
            rows.append({"file": f"{cls}/{f}", "class": cls, "source_url": f"http://x/{f}", "textilenet_file": f,
                         "width": 200, "height": 200, "md5": f"m{cls}{i}", "status": "candidate"})
    with open(cand / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dec = tmp_path / "decisions.csv"
    with open(dec, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "decision", "description"])
        w.writeheader()
        for stem, d in (("cotton_000", "keep"), ("cotton_001", "drop"), ("cotton_002", "keep"), ("cotton_003", "keep"),
                        ("silk_000", "keep")):
            w.writerow({"stem": stem, "decision": d, "description": f"desc {stem}"})
    return cand, dec


def test_export_unseen_keeps_caps_and_records(tmp_path):
    cand, dec = _setup(tmp_path)
    out = tmp_path / "unseen_data"
    counts = eu.export(cand, dec, out, per_class=2)
    assert counts == {"Cotton": 2, "Silk": 1}
    with open(out / "sources.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["file"] for r in rows] == ["Cotton/cotton_000.jpg", "Cotton/cotton_002.jpg", "Silk/silk_000.jpg"]
    assert {"file", "class", "source_url", "textilenet_file", "md5", "description", "label_source", "screened_by"} <= set(rows[0])
    assert rows[0]["label_source"] == "textilenet" and rows[0]["screened_by"] == "claude"
    assert (out / "Cotton" / "cotton_002.jpg").exists() and not (out / "Cotton" / "cotton_001.jpg").exists()
    # silk_001 was never screened -> not exported
    assert not (out / "Silk" / "silk_001.jpg").exists()


def test_export_unseen_rerun_replaces(tmp_path):
    cand, dec = _setup(tmp_path)
    out = tmp_path / "unseen_data"
    eu.export(cand, dec, out, per_class=3)
    counts = eu.export(cand, dec, out, per_class=1)
    assert counts == {"Cotton": 1, "Silk": 1}
    assert len(list((out / "Cotton").glob("*.jpg"))) == 1
