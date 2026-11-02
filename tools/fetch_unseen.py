"""ดาวน์โหลดรูปผู้สมัครสำหรับชุด Unseen จาก TextileNet (รายการ URL ใน json/fibre_{test,train}.json)

รัน: .venv\\Scripts\\python.exe -I tools\\fetch_unseen.py --meta <dir ที่มี fibre_test.json> --out <dir> [--per-class 300]
- สุ่มรายการของ 4 คลาส (seed คงที่) ใช้ชุด test ของ TextileNet ก่อน ถ้าไม่พอค่อยเติมจากชุด train
- คัดกรองอัตโนมัติ: เปิดเป็นรูปได้, ด้านสั้น >= 128 px, md5 ไม่ซ้ำกันเองและไม่ซ้ำกับ dataset เดิม (results/manifest.csv)
- เขียน candidates.csv (file, class, source_url, width, height, md5, status) ให้หน้ารีวิวใช้ต่อ
ไฟล์ที่ดาวน์โหลดเป็นข้อมูลภายนอก: เปิดด้วย PIL เพื่อตรวจเท่านั้น ไม่รันอะไรจากไฟล์
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

CLASS_MAP = {"cotton": "Cotton", "polyester": "Polyester", "silk": "Silk", "viscose_rayon": "Viscose"}
MIN_SIDE = 128
MAX_BYTES = 8 * 2**20


def fetch(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (research; FabricSort student project)"})
        if r.ok and 0 < len(r.content) <= MAX_BYTES:
            return r.content
    except requests.RequestException:
        pass
    return None


def check(data: bytes):
    """-> (w, h, ext) ถ้าเป็นรูปที่ใช้ได้ ไม่เช่นนั้น None"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            fmt = (im.format or "").lower()
    except Exception:
        return None
    if min(w, h) < MIN_SIDE or fmt not in ("jpeg", "png", "webp"):
        return None
    return w, h, {"jpeg": ".jpg"}.get(fmt, "." + fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "manifest.csv")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--classes", nargs="*", default=list(CLASS_MAP.values()), help="เช่น Silk")
    ap.add_argument("--append", action="store_true", help="ต่อจาก candidates.csv เดิม (ข้าม URL ที่เคยลอง)")
    args = ap.parse_args()

    known = set()
    if args.manifest.exists():
        with open(args.manifest, encoding="utf-8") as f:
            known = {row["md5"] for row in csv.DictReader(f) if row.get("md5")}
    rng = random.Random(42)
    rows, seen, used = [], set(known), set()
    old_rows = []
    if args.append and (args.out / "candidates.csv").exists():
        with open(args.out / "candidates.csv", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        seen |= {r["md5"] for r in old_rows}
        used = {r["source_url"] for r in old_rows}
    for tn_cls, cls in CLASS_MAP.items():
        if cls not in args.classes:
            continue
        pool = []
        for part in ("fibre_test", "fibre_train"):
            items = [d for d in json.loads((args.meta / f"{part}.json").read_text(encoding="utf-8")) if d[1] == tn_cls]
            rng.shuffle(items)
            pool += items
        pool = [d for d in pool if d[2] not in used]
        (args.out / cls).mkdir(parents=True, exist_ok=True)
        start = sum(r["class"] == cls for r in old_rows)
        kept, i = 0, 0
        with ThreadPoolExecutor(args.workers) as ex:
            while kept < args.per_class and i < len(pool):
                batch = pool[i:i + 4 * args.workers]
                i += len(batch)
                for (fname, _, url), data in zip(batch, ex.map(lambda d: fetch(d[2]), batch)):
                    if kept >= args.per_class:
                        break
                    if data is None:
                        continue
                    info = check(data)
                    md5 = hashlib.md5(data).hexdigest()
                    if info is None or md5 in seen:
                        continue
                    seen.add(md5)
                    w, h, ext = info
                    name = f"{cls.lower()}_{start + kept:03d}{ext}"
                    (args.out / cls / name).write_bytes(data)
                    rows.append({"file": f"{cls}/{name}", "class": cls, "source_url": url, "textilenet_file": fname,
                                 "width": w, "height": h, "md5": md5, "status": "candidate"})
                    kept += 1
        print(f"{cls}: เก็บ {kept} รูป (ลองไป {i} URL)", flush=True)
    with open(args.out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list((rows or old_rows)[0]))
        w.writeheader()
        w.writerows(old_rows + rows)
    print("บันทึก", args.out / "candidates.csv", len(rows), "รูป")


if __name__ == "__main__":
    main()
