# FabricSort: ระบบจำแนกเศษผ้าเพื่อนำไปรีไซเคิล

จำแนกรูปเศษผ้า 1 รูปเป็น **Cotton / Polyester / Silk / Viscose** พร้อมค่าความมั่นใจ (งานวิชา Applied Machine Learning)
รายละเอียดความต้องการอยู่ใน [PRD](docs/PRD%20ระบบจำแนกเศษผ้าเพื่อนำไปรีไซเคิล%20(FabricSort).md)
ส่วน design และแผนของเวอร์ชันนี้อยู่ใน `docs/superpowers/specs/` และ `docs/superpowers/plans/`

เดโมออนไลน์: https://fabricsort.streamlit.app/

## เริ่มบนเครื่องใหม่

repo นี้มีเฉพาะสิ่งที่ต้องใช้เปิดและรันโค้ด

| มีใน repo | ไม่มีใน repo (ต้องเตรียมเอง) |
|---|---|
| โค้ด, notebook 2 ไฟล์, แอป, tests, tools | `dataset/` รูปผ้า (ดาวน์โหลดเอง ดูด้านล่าง) |
| `results/` ผลทั้งหมด (CSV, JSON, กราฟ, `optuna.db`, `split.csv`) | `models/runs/` weights ของทุก run (2.1 GB) |
| `models/best_model.pth` โมเดลที่แอปใช้ (EfficientNet-B0 หลังจูน, 16 MB) | รูปใน `unseen_data/` (ดาวน์โหลดใหม่ด้วยสคริปต์) |
| `data_audit/` ผลตรวจคุณภาพ TextileClass-7 | `.venv/` |

**1. ติดตั้ง** (Python 3.12)
```bash
git clone https://github.com/Natthaphatkaewkao/finalProMl.git
cd finalProMl
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
ถ้าเครื่องไม่มี GPU NVIDIA ให้แก้ `requirements.txt` เป็น `torch==2.14.0`, `torchvision==0.29.0` และเปลี่ยน index เป็น `https://download.pytorch.org/whl/cpu`

**2. เปิดแอป (ไม่ต้องมี dataset)**
```bash
streamlit run app.py
```
ทั้ง 4 หน้าใช้ผลใน `results/` และโมเดลใน `models/` ได้ทันที

**3. รัน notebook ใหม่ (ต้องมี dataset)**
- **TextileClass-7:** Mendeley Data DOI [10.17632/m2h53zr39s.2](https://doi.org/10.17632/m2h53zr39s.2) (CC BY 4.0) แตกไฟล์เป็น `dataset/TextileClass7/<Class>/...`
- **iBUG Fabrics:** ขอดาวน์โหลดจาก Imperial College London (กรอกฟอร์ม, ใช้เพื่อการวิจัยเท่านั้น ห้ามเผยแพร่ต่อ) แตกไฟล์เป็น `dataset/iBUG/<Class>/<pieceID>/im_*.png`
- `<Class>` = `Cotton`, `Polyester`, `Silk`, `Viscose`
- `results/split.csv` ทำให้แบ่ง Train/Val/Test ได้เหมือนเดิมทุกรูป เมื่อไม่มี `models/runs/` notebook จะเทรนทุก run ใหม่ (ประมาณ 5–6 ชั่วโมงบน RTX 4050)
- รูป Unseen: `python tools/download_unseen.py` (โหลดจากลิงก์ใน `unseen_data/sources.csv` และตรวจ md5 ลิงก์บางอันอาจเสียไปแล้ว)
- ลำดับ: `fabric_classification.ipynb` → `tune_others.ipynb` (จูน DenseNet121, EfficientNet-B0) → `python tools/plot_before_after_4models.py` → `python tools/export_deploy_model.py` (ให้แอปใช้ EfficientNet-B0 ตามเดิม เพราะ notebook ส่วนที่ 12 จะ export ResNet50)

**4. ทดสอบ:** `pytest tests`

## Flow ของเวอร์ชันนี้

1. **คัดเลือก** CNN 4 ตัวจาก torchvision ด้วยสูตรเทรนมาตรฐานเดียวกัน: ResNet50, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny
2. **เลือก Top 1** จาก Val Macro-F1 (ต่างกันไม่เกิน 0.01 เลือกตัวที่ inference เร็วกว่า)
3. **ก่อนจูน:** Top 1 สูตร default 3 seed → **จูน** hyperparameter ด้วย Optuna 15 trial → **หลังจูน:** สูตรที่ดีที่สุด 3 seed
4. ประเมินบนชุด Test ครั้งเดียว (ก่อน vs หลัง: mean ± std, รายคลาส, แยกแหล่ง, bootstrap CI, McNemar) แล้ว export โมเดล
5. แอป Streamlit 3 หน้า: เปรียบเทียบ 4 โมเดล / ก่อน vs หลังจูน / ทำนายรูป

ถ้า Test Macro-F1 หลังจูนต่ำกว่าเป้า 0.85 (PRD) ขั้นต่อไปคือลอง ConvNeXt V2-Tiny (timm) ด้วย flow เดียวกัน

**ทำไมไม่ใช้ Keras:** แผนแรกเลือกโมเดลจาก Keras Applications แต่ Keras 3 (backend PyTorch) บน RTX 4050 6 GB
ใช้หน่วยความจำเกิน 6 GB ที่ batch 32, หน่วยความจำรั่วทุก step ที่ batch 16 และช้ากว่ามาก (EfficientNetB0 417 วินาที/epoch,
ConvNeXtTiny 523 วินาที/epoch เทียบกับ torchvision 42–61 วินาที/epoch) ส่วน TensorFlow ใช้ GPU บน Windows ไม่ได้
จึงใช้สถาปัตยกรรมเดียวกันจาก torchvision

## ไฟล์ในโปรเจค

| ไฟล์ | คือ |
|---|---|
| `fabric_classification.ipynb` | notebook หลัก (สร้างด้วย `tools/build_notebook.py`) |
| `fabric_lib.py` | โค้ดที่ notebook กับแอปใช้ร่วมกัน (สร้างจาก notebook ส่วน 1.1) |
| `app.py` | แอป Streamlit (สร้างจาก notebook ส่วน 13 ต้นฉบับอยู่ที่ `app_src/app.py`) |
| `tools/` | `build_notebook.py` (สร้าง notebook), `tuning_helpers.py` (search space, เกณฑ์เลือก, CI, McNemar) |
| `tests/` | ทดสอบด้วย `pytest` |
| `models/` | `best_model.pth` + `model_meta.json`, `runs/<run_id>.pth` (ทุก run) |
| `results/` | `manifest.csv`, `split.csv`, `runs.csv`, `screening.csv`, `selection.json`, `optuna.db`, `optuna_trials.csv`, `test_before_after.csv` ฯลฯ และกราฟ |
| `data_audit/` | ผลตรวจคุณภาพ TextileClass-7 (`fabric_groups.csv`, `exclude.csv`) |

## 1. ติดตั้ง

ต้องการ Python 3.11+ และ GPU NVIDIA (แนะนำ)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

บรรทัดสุดท้ายควรพิมพ์ `True NVIDIA GeForce RTX 4050 Laptop GPU`

### ปัญหา Smart App Control

ถ้าขึ้น `An Application Control policy has blocked this file`, `WinError 4551` หรือ `The scipy install you are using seems to be broken` แปลว่า Windows 11 **Smart App Control** เปิดอยู่ในโหมดบังคับใช้ และบล็อกไฟล์ DLL ของไลบรารี Python (torch, scipy, scikit-learn) ที่ไม่ได้ลงลายเซ็นดิจิทัล ไม่ใช่ปัญหาของโค้ด ติดตั้งใหม่ก็ไม่หาย

ตรวจสถานะ (0 = ปิด, 1 = เปิด, 2 = โหมดประเมิน):
```bash
powershell -c "(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState"
```

ทางแก้ (เลือกหนึ่งทาง)
1. **ปิด Smart App Control:** Windows Security → App & browser control → Smart App Control settings → Off
   ข้อควรรู้: ใน Windows 11 หลายรุ่น เมื่อปิดแล้วจะเปิดกลับไม่ได้ถ้าไม่รีเซ็ต Windows
2. **รันใน WSL2 (Ubuntu):** Smart App Control ไม่มีผลกับโปรแกรม Linux และ WSL2 ใช้ GPU NVIDIA ได้ เปิดโฟลเดอร์โปรเจคจาก WSL ใน VS Code (`\\wsl$` หรือ `/mnt/c/...`) แล้วสร้าง venv ใน WSL
3. **รันบนเครื่องของสมาชิกคนอื่น** ที่ Smart App Control ปิดอยู่หรืออยู่ในโหมดประเมิน

## 2. เตรียมข้อมูล

วางรูปตามโครงสร้างนี้ (ชื่อโฟลเดอร์คลาสไม่สนตัวพิมพ์เล็ก/ใหญ่ รูปอยู่ในโฟลเดอร์ย่อยได้)

```
dataset/
  iBUG/{Cotton,Polyester,Silk,Viscose}/...
  TextileClass7/{Cotton,Polyester,Silk,Viscose}/...
```

- **iBUG Fabrics** (Kampouris et al., ECCV 2016, Imperial College London): https://ibug.doc.ic.ac.uk/resources/fabrics/ ใช้เพื่อการวิจัยเท่านั้น
- **TextileClass-7**: Mendeley Data, Version 2

TextileClass-7 ให้แตกไฟล์เฉพาะ 4 คลาสลง `dataset/TextileClass7/<คลาส>/` แบบไม่มีโฟลเดอร์ย่อย (ไม่ใช้ train/valid/test เดิมของ Roboflow) ส่วน iBUG ให้คงโครงสร้าง `dataset/iBUG/<คลาส>/<ID ชิ้น>/im_1..im_4.png`

ถ้าชิ้นผ้าเดียวกันมีหลายรูป (iBUG ถ่ายแต่ละชิ้นภายใต้แสง 4 ทิศ) รูปของชิ้นนั้นต้องอยู่โฟลเดอร์ย่อยเดียวกัน หรือตั้ง `GROUP_REGEX` ใน cell Config เพื่อไม่ให้ภาพชิ้นเดียวกันรั่วข้ามชุด Train/Test

## 3. รัน

เปิด `fabric_classification.ipynb` จากโฟลเดอร์โปรเจค เลือก kernel ของ `.venv` แล้วกด **Run All**

- เวลาโดยประมาณบน RTX 4050: คัดเลือก ~1.5 ชม., ก่อนจูน ~50 นาที, Optuna ~2 ชม., หลังจูน ~1–1.5 ชม. (รวม ~5–6 ชม. รันข้ามคืนได้)
- ทุก run ถูกบันทึกไว้ ถ้าหยุดกลางทาง กด Run All ใหม่จะทำต่อจากจุดเดิม (Optuna ต่อจาก `results/optuna.db`)
- ทดสอบว่ารันครบทุก cell ได้ก่อนรันจริง: ตั้ง `SMOKE = True` ใน Config (2 epoch, 2 trial, ข้อมูลส่วนน้อย ใช้เวลาไม่กี่นาที)
  ผลจะเขียนแยกที่ `results/_smoke/` และ `models/_smoke/`

แอปเปิดที่ http://localhost:8501 เมื่อจบ notebook หรือรันเองได้ด้วย

```bash
streamlit run app.py
```

ทดสอบโค้ด: `python -m pytest tests -q`
