"""ดาวน์โหลดรูปชุด Unseen กลับมาตามรายการใน unseen_data/sources.csv (ใช้บนเครื่องใหม่ที่ไม่มีรูป)

รัน: python tools/download_unseen.py
- โหลดจาก source_url ไปเก็บที่ unseen_data/<file> แล้วตรวจ md5 ให้ตรงกับรูปเดิม
- ข้ามรูปที่มีอยู่แล้ว · รูปที่ลิงก์เสียหรือ md5 ไม่ตรงจะถูกรายงาน (เว็บต้นทางอาจลบหรือเปลี่ยนรูปไปแล้ว)
ไฟล์ที่ดาวน์โหลดเป็นข้อมูลภายนอก: เปิดด้วย PIL เพื่อตรวจเท่านั้น ไม่รันอะไรจากไฟล์
"""
import csv
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

UNS = Path(__file__).resolve().parents[1] / "unseen_data"


def fetch(row):
    out = UNS / row["file"]
    if out.exists():
        return "มีอยู่แล้ว"
    try:
        r = requests.get(row["source_url"], timeout=30)
        r.raise_for_status()
        data = r.content
        Image.open(io.BytesIO(data)).verify()
    except Exception as e:
        return f"โหลดไม่ได้ ({type(e).__name__})"
    if hashlib.md5(data).hexdigest() != row["md5"]:
        return "md5 ไม่ตรง (รูปต้นทางเปลี่ยน)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return "ok"


def main():
    rows = list(csv.DictReader(open(UNS / "sources.csv", encoding="utf-8")))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, rows))
    for k in sorted(set(res)):
        print(f"{k}: {res.count(k)}")
    for row, r in zip(rows, res):
        if r not in ("ok", "มีอยู่แล้ว"):
            print("  ", row["file"], r)


if __name__ == "__main__":
    main()
