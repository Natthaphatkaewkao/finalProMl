"""รวมรูปผู้สมัคร Unseen เป็นแผ่นละ 24 รูป (6×4) พร้อมเลขกำกับ ใช้ตอนคัดกรองทีละรูป

รัน: .venv\\Scripts\\python.exe -I tools\\contact_sheet.py --candidates <dir ที่มี candidates.csv หรือโฟลเดอร์คลาส> --out <dir> --cls Cotton
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

COLS, ROWS, CELL = 6, 4, 220


def sheets(files, out: Path, prefix: str):
    out.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    paths = []
    for s in range(0, len(files), per):
        sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, f in enumerate(files[s:s + per]):
            x, y = (k % COLS) * CELL, (k // COLS) * (CELL + 22)
            try:
                with Image.open(f) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((CELL - 6, CELL - 6))
                    sheet.paste(im, (x + (CELL - im.width) // 2, y + 22 + (CELL - im.height) // 2))
            except Exception:
                draw.text((x + 10, y + 100), "unreadable", fill="red")
            draw.text((x + 6, y + 4), f.stem, fill="black")
        p = out / f"{prefix}_{s // per:02d}.jpg"
        sheet.save(p, quality=85)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cls", required=True)
    a = ap.parse_args()
    files = sorted((a.candidates / a.cls).glob("*.*"))
    for p in sheets(files, a.out, a.cls.lower()):
        print(p)
