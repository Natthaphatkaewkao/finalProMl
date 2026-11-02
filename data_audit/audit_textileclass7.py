"""ตรวจคุณภาพ TextileClass-7: หารูปของผ้าชิ้นเดียวกันและ label ที่ขัดแย้งกัน

รัน:  python data_audit/audit_textileclass7.py   (จากโฟลเดอร์โปรเจค ใช้ GPU ถ้ามี ประมาณ 1 นาที)

วิธี
1. ฝังทุกรูปด้วย ResNet50 pretrained ImageNet (feature 2048 มิติก่อนชั้นสุดท้าย) แล้วเฉลี่ยต่อรูปต้นฉบับ
   (Roboflow ทำ augmentation 3 เวอร์ชันต่อรูป ชื่อ "<ต้นฉบับ>_jpg.rf.<hash>.jpg")
2. สองรูปถือว่าเป็นผ้าชิ้นเดียวกัน เมื่อ cosine similarity >= 0.90 และถ่ายจากเครื่องเดียวกันห่างกันไม่เกิน 10 วินาที
   (หรือเลขไฟล์ห่างกันไม่เกิน 3 สำหรับชื่อไฟล์แบบเลขลำดับ) หรือเมื่อ similarity >= 0.95 โดยไม่ต้องมีหลักฐานเวลา
   ชื่อไฟล์บอกเครื่องและเวลา: IMG20260609200611 / IMG_20260523_123447 / IMG-20260609-WA0176 / IMG_5943_result
3. คู่ที่อยู่คลาสเดียวกัน -> รวมเป็นกลุ่มเดียว (fabric_groups.csv) เพื่อไม่ให้รั่วข้าม Train/Test
   คู่ที่อยู่คนละคลาส -> label ขัดแย้ง เขียนลง review_pairs.csv + review_pairs.png ให้คนตรวจ
   รายการที่ตัดสินใจตัดออกจริงเก็บใน exclude.csv (แก้ด้วยมือ ไม่ถูก script นี้เขียนทับ)

ผลลัพธ์
- fabric_groups.csv : photo_group -> fabric_id   (notebook ใช้แทน sample_id)
- review_pairs.csv/.png : คู่ต่างคลาสที่ต้องตรวจ (A = หลักฐานเวลา+ภาพ, B = ภาพคล้ายมาก sim >= 0.92)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import fabric_lib as fl  # noqa: E402

DATA = PROJECT / "dataset"
OUT = Path(__file__).resolve().parent
SIM_LINK, SIM_ANY, SIM_REVIEW = 0.90, 0.95, 0.92
MAX_DT, MAX_GAP = 10, 3
GROUP_RX = r"_(jpe?g|png)\.rf\.[0-9a-f]+$"


def embed(paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    tf = fl.build_transforms(224, train=False)
    out = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for i in range(0, len(paths), 64):
            x = torch.stack([tf(fl.load_pil(DATA / p)) for p in paths[i:i + 64]]).to(device)
            out.append(torch.nn.functional.normalize(net(x).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def device_and_order(stem):
    """(เครื่อง, ลำดับ) จากชื่อไฟล์: เวลาเป็นวินาทีสำหรับชื่อแบบ timestamp, เลขไฟล์สำหรับชื่อแบบลำดับ"""
    if m := re.match(r"IMG(\d{8})(\d{6})", stem):
        return "phoneA", pd.Timestamp(f"{m[1]} {m[2]}").value / 1e9, MAX_DT
    if m := re.match(r"IMG_(\d{8})_(\d{6})", stem):
        return "phoneB", pd.Timestamp(f"{m[1]} {m[2]}").value / 1e9, MAX_DT
    if m := re.match(r"IMG-(\d{8})-WA(\d{4})", stem):
        return f"whatsapp{m[1]}", float(m[2]), MAX_GAP
    if m := re.match(r"IMG_(\d{4})", stem):
        return "phoneC", float(m[1]), MAX_GAP
    return None, np.nan, 0


def main():
    files = pd.DataFrame([{"path": p.relative_to(DATA).as_posix(), "class": c,
                           "group": f"TextileClass7/{c}/" + re.sub(GROUP_RX, "", p.stem)}
                          for c in fl.CLASSES for p in sorted((DATA / "TextileClass7" / c).iterdir())])
    E = embed(files.path.tolist())
    gb = files.groupby("group", sort=False)
    g = pd.DataFrame({"group": list(gb.groups), "class": gb["class"].first().values,
                      "rep_path": gb["path"].first().values})
    G = np.stack([E[ix].mean(0) for ix in gb.indices.values()])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    S = G @ G.T
    dev = g.group.str.split("/").str[-1].map(device_and_order)
    g["device"], g["order"], g["tol"] = dev.str[0], dev.str[1], dev.str[2]

    i, j = np.triu_indices(len(g), 1)
    keep = S[i, j] >= min(SIM_LINK, SIM_REVIEW)
    i, j, s = i[keep], j[keep], S[i, j][keep]
    same_dev = (g.device.values[i] == g.device.values[j]) & pd.notna(g.device.values[i])
    adjacent = same_dev & (np.abs(g.order.values[i] - g.order.values[j]) <= g.tol.values[i])
    linked = (adjacent & (s >= SIM_LINK)) | (s >= SIM_ANY)
    cross = g["class"].values[i] != g["class"].values[j]

    # รวมกลุ่มผ้าชิ้นเดียวกัน (คลาสเดียวกัน) ด้วย union-find
    parent = list(range(len(g)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(i[linked & ~cross], j[linked & ~cross]):
        parent[find(a)] = find(b)
    g["fabric_id"] = [g.group[find(k)] for k in range(len(g))]
    g[["group", "fabric_id"]].rename(columns={"group": "photo_group"}).to_csv(OUT / "fabric_groups.csv", index=False)
    print(f"photo groups {len(g)} -> fabric groups {g.fabric_id.nunique()}")

    # คู่ต่างคลาสที่ต้องตรวจ
    rev = pd.DataFrame({"a": g.group.values[i], "b": g.group.values[j], "sim": s,
                        "kind": np.where(linked & adjacent, "A", "B")})[cross & ((linked & adjacent) | (s >= SIM_REVIEW))]
    rev = rev.sort_values(["kind", "sim"], ascending=[True, False]).reset_index(drop=True)
    rev["id"] = rev.groupby("kind").cumcount().add(1).astype(str).radd(rev.kind)
    cls = g.set_index("group")["class"]
    rev["class_a"], rev["class_b"] = rev.a.map(cls).values, rev.b.map(cls).values
    rev[["id", "kind", "sim", "a", "class_a", "b", "class_b"]].to_csv(OUT / "review_pairs.csv", index=False)
    render(rev, g.set_index("group"))
    print(f"review pairs: {len(rev)} ({rev.kind.value_counts().to_dict()})")


def render(rev, g, T=180, cols=3):
    font, bold = ImageFont.load_default(14), ImageFont.load_default(15)
    W, H = 2 * T + 30, T + 58
    sheet = Image.new("RGB", (cols * W, max(1, (len(rev) + cols - 1) // cols) * H + 10), "white")
    d = ImageDraw.Draw(sheet)
    for n, r in rev.iterrows():
        x0, y0 = (n % cols) * W + 8, (n // cols) * H + 8
        d.text((x0, y0), f"{r.id}  sim {r.sim:.3f}", fill="black", font=bold)
        for k, grp in enumerate((r.a, r.b)):
            sheet.paste(Image.open(DATA / g.loc[grp, "rep_path"]).convert("RGB").resize((T, T)), (x0 + k * (T + 6), y0 + 20))
            d.text((x0 + k * (T + 6), y0 + 22 + T), f"{g.loc[grp, 'class']}  {grp.split('/')[-1][:20]}", fill="black", font=font)
    sheet.save(OUT / "review_pairs.png")


if __name__ == "__main__":
    main()
