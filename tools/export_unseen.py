"""รวมผลคัดกรอง (decisions.csv) กับรูปผู้สมัคร (candidates.csv) -> unseen_data/<Class>/ + sources.csv

รัน: .venv\\Scripts\\python.exe -I tools\\export_unseen.py --candidates <dir> --decisions <decisions.csv> [--out unseen_data] [--per-class 100]
- เก็บเฉพาะรูปที่ decision = keep เรียงตามลำดับไฟล์ ไม่เกิน per_class รูปต่อคลาส
- label มาจาก TextileNet (label_source=textilenet) ส่วนการคัดกรองทำโดย Claude (screened_by=claude)
- รันซ้ำได้: ลบโฟลเดอร์ปลายทางของแต่ละคลาสแล้วเขียนใหม่ทั้งหมด
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

FIELDS = ["file", "class", "source_url", "textilenet_file", "md5", "description", "label_source", "screened_by"]


def export(candidates: Path, decisions: Path, out: Path, per_class: int = 100) -> dict[str, int]:
    with open(decisions, encoding="utf-8") as f:
        dec = {r["stem"]: r for r in csv.DictReader(f)}
    with open(candidates / "candidates.csv", encoding="utf-8") as f:
        cand = sorted(csv.DictReader(f), key=lambda r: r["file"])
    out.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {}
    for r in cand:
        cls, stem = r["class"], Path(r["file"]).stem
        d = dec.get(stem)
        if d is None or d["decision"] != "keep" or counts.get(cls, 0) >= per_class:
            continue
        if cls not in counts:
            shutil.rmtree(out / cls, ignore_errors=True)
            (out / cls).mkdir(parents=True)
            counts[cls] = 0
        shutil.copy2(candidates / r["file"], out / r["file"])
        counts[cls] += 1
        rows.append({"file": r["file"], "class": cls, "source_url": r["source_url"], "textilenet_file": r["textilenet_file"],
                     "md5": r["md5"], "description": d["description"], "label_source": "textilenet", "screened_by": "claude"})
    with open(out / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "unseen_data")
    ap.add_argument("--per-class", type=int, default=100)
    a = ap.parse_args()
    print(export(a.candidates, a.decisions, a.out, a.per_class))
