# FabricSort v3: คัดเลือก CNN จาก Keras → จูน Top 1 → Streamlit

วันที่: 2026-10-30 · สถานะ: รอผู้ใช้ review

## 1. เป้าหมาย

ปรับการเทรนโมเดลจำแนกเศษผ้า (Cotton / Polyester / Silk / Viscose) ใหม่ตาม flow:

1. เลือก CNN 4 ตัวจาก **Keras Applications**
2. เทรนทั้ง 4 ด้วยสูตรมาตรฐานเดียวกัน วัดผล แล้วเลือก Top 1 (Val Macro-F1)
3. Fine-tune / จูน hyperparameter ของ Top 1 และวัดผล 2 จุด: **ก่อน** (สูตร default) และ **หลัง** (สูตรที่จูนแล้ว)
4. ทั้งหมดอยู่ใน `fabric_classification.ipynb` (Run All ครั้งเดียว)
5. แอป Streamlit สำหรับ visualization 3 หน้า

**เกณฑ์สำเร็จ:** notebook รันจบโดยไม่ต้องแก้ cell นอกจาก Config; มีตารางเปรียบเทียบ 4 โมเดล, ผลก่อน/หลังจูน (mean ± std 3 seed) บน Val และ Test; แอปเปิดได้ทั้ง 3 หน้าและทำนายรูปได้ < 1 วินาที เป้าคุณภาพตาม PRD คือ Test Macro-F1 ≥ 0.85 (ถ้าไม่ถึงให้ทำขั้น 6 ในส่วน 3)

**ข้อจำกัด:** RTX 4050 6 GB, Windows 11 (Smart App Control ปิดแล้ว), `.venv` ของโปรเจค (Python 3.14, torch 2.14 + CUDA) งบเวลาเทรนรวม ~5–6 ชม. (รันข้ามคืน)

## 2. ข้อมูล

- ใช้ `results/split.csv` เดิม (Train 4,887 / Val 326 / Test 328; Val/Test 1 รูปต่อชิ้นผ้า) และ `data_audit/` เดิม ส่วน 0–3 ของ notebook (Config, Setup, Load+EDA, Clean+Split) คงตรรกะเดิม
- ตรวจ data fingerprint ก่อนเทรน: checkpoint ใช้ซ้ำได้เฉพาะเมื่อ fingerprint ตรง
- **Test ใช้ครั้งเดียวตอนท้าย** (ข้อยกเว้นเดียวคือขั้น 6 ด้านล่าง ซึ่งบันทึกไว้ใน `selection.json`)

## 3. Flow การเทรน

**Framework:** Keras 3 + `KERAS_BACKEND="torch"` (ใช้ GPU ผ่าน PyTorch CUDA) mixed precision `mixed_float16`, batch 32 (OOM → 16 อัตโนมัติ)

**โมเดล (Keras Applications, weights ImageNet):** `ResNet50V2`, `DenseNet121`, `EfficientNetB0`, `ConvNeXtTiny`

**โครงโมเดล:** `Input(224,224,3, ค่า 0–255)` → `preprocess_input` ของแต่ละโมเดล (อยู่ในโมเดล) → base (`training=False` เพื่อให้ BatchNorm อยู่ใน inference mode) → GlobalAveragePooling → Dropout → Dense(4, softmax, float32)

**สูตร default (= ผล "ก่อน"):** 2 เฟสแบบ Keras
- เฟส 1: แช่แข็ง base, เทรนหัว 3 epoch, Adam(W) lr 1e-3
- เฟส 2: เปิด 30% ของ layer บนสุด, lr 1e-5 + cosine decay, สูงสุด 30 epoch, early stopping patience 7 บน Val Macro-F1 แล้วคืน weights ที่ดีที่สุด
- loss: categorical cross-entropy ถ่วงน้ำหนักตามคลาส, label smoothing 0, augmentation พื้นฐาน (random resized crop, หมุน 90°, flip, color jitter), ไม่มี MixUp

| ขั้น | งาน | seed | ประมาณเวลา* |
|---|---|---|---|
| 1 คัดเลือก | เทรน 4 โมเดลด้วยสูตร default | 42 | ~1.5 ชม. |
| 2 เลือก Top 1 | Val Macro-F1 สูงสุด; ต่างกัน ≤ 0.01 → inference เร็วกว่า | – | – |
| 3 ก่อน | Top 1 สูตร default เพิ่ม 2 seed | 7, 2024 (+42 จากขั้น 1) | ~50 นาที |
| 4 จูน | Optuna TPE 15 trial, trial ละ 12 epoch (เฟส 2), MedianPruner, objective = Val Macro-F1 | 42 | ~2 ชม. |
| 5 หลัง | Top 1 สูตรที่ดีที่สุดจาก Optuna เต็ม 30 epoch | 42, 7, 2024 | ~1–1.5 ชม. |
| 6 สำรอง (มีเงื่อนไข) | ถ้า Test Macro-F1 เฉลี่ยของ "หลัง" < 0.85 → เทรน ConvNeXt V2-Tiny (timm/PyTorch) ด้วย flow เดียวกัน (default + จูน); เลือกระหว่าง Keras Top 1 กับตัวนี้ด้วย **Val** | 42, 7, 2024 | +~3 ชม. |

\* ประมาณจาก PyTorch เดิม ต้องยืนยันด้วย spike (ส่วน 7) ถ้า Keras ช้ากว่ามากจะลดจำนวน trial และแจ้งผู้ใช้ก่อน

**Search space ของ Optuna:** lr หัว 1e-4–3e-3 (log), lr fine-tune 1e-6–1e-4 (log), สัดส่วน layer ที่เปิด 0.1–1.0, epoch เฟส 1: 1–5, dropout 0–0.5, weight decay 1e-5–0.05 (log), label smoothing 0–0.15, augmentation {base, strong}, MixUp/CutMix {on, off}

**ตัวชี้วัดที่บันทึกทุก run:** Val Accuracy, Macro Precision/Recall/F1, F1 และ Recall รายคลาส, Recall ต่ำสุด, เวลาเทรน, จำนวนพารามิเตอร์, history ราย epoch; สำหรับโมเดลที่ใช้ตัดสินใจ: เวลา inference ต่อรูป (ค่ามัธยฐาน 50 รูป, end-to-end)

**การประเมิน Test (ครั้งเดียว):** ก่อน vs หลัง: Macro-F1 / Accuracy mean ± std 3 seed, รายคลาส, แยกแหล่ง (iBUG / TextileClass7), confusion matrix, bootstrap 95% CI (1,000 รอบ), McNemar (seed ที่ดีที่สุดบน Val ของแต่ละฝั่ง), เทียบเป้า PRD

**Export:** seed ที่ Val Macro-F1 สูงสุดของ "หลัง" → `models/best_model.keras` + `model_meta.json`

## 4. ไฟล์

**สำรองก่อนเขียนทับ:** คัดลอก `fabric_classification.ipynb`, `fabric_lib.py`, `app.py`, `README.md`, `requirements.txt`, `models/`, `results/` ไป `backup_2026-10-30/` ไฟล์เดิมที่ไม่ใช้แล้ว (`models/candidates/`, `models/best_model.pth`, กราฟ SVM/KNN/DINO ฯลฯ) ย้ายเข้า backup ไม่ปนกับผลใหม่ เก็บ `results/manifest.csv`, `results/split.csv` ไว้ที่เดิม

**เขียนทับ:**

| ไฟล์ | เนื้อหาใหม่ |
|---|---|
| `fabric_classification.ipynb` | ส่วน: ติดตั้ง → 0 Config → 1 Setup (เขียน `fabric_lib.py`) → 2 Load+EDA → 3 Clean+Split → 4 ฟังก์ชันเทรน → 5 คัดเลือก 4 โมเดล → 6 เลือก Top 1 → 7 ก่อน (3 seed) → 8 Optuna → 9 หลัง (3 seed) → 10 Test ก่อน vs หลัง (+ ขั้นสำรอง) → 11 Error analysis → 12 Export → 13 เขียน `app.py` → 14 เปิดแอป |
| `fabric_lib.py` | `CLASSES`, `load_pil`, augmentation, `KERAS_ZOO` (4 โมเดล + preprocess), `build_model`, `KerasPredictor`, `load_predictor` |
| `app.py` | แอป 3 หน้า |
| `README.md`, `requirements.txt` | อัปเดตตาม flow ใหม่ (+ `keras`, `optuna`, `plotly`) |

**ผลลัพธ์:**
- `results/`: `runs.csv` (1 แถว/run: stage, model, seed, params, metrics, เวลา), `screening.csv`, `selection.json`, `optuna.db`, `optuna_trials.csv`, `optuna_importance.csv`, `histories/<run_id>.json`, `test_before_after.csv`, `per_class_test.csv`, `per_source_test.csv`, `significance.csv`, `confusion_{before,after}.json`, กราฟ PNG
- `models/`: `runs/<run_id>.keras` (run_id = hash ของ model + params + seed + epochs + fingerprint), `best_model.keras`, `model_meta.json` (`framework: "keras"`, ชื่อโมเดล, คลาส, img_size, params, metrics)

**ไม่แตะ:** `fabric_classification_old.ipynb`, `backup_original_2026-10-29/`, `dataset/`, `data_audit/`

## 5. โครงโค้ดใน notebook

- `train_one(arch, params, seed, max_epochs, trial=None) -> run_id`: ฟังก์ชันเทรนเดียวของทุกขั้น ถ้ามีไฟล์ของ run_id อยู่แล้ว (และ `SKIP_TRAINING_IF_EXISTS`) โหลดผลเดิม ถ้ามี `trial` รายงาน Val Macro-F1 ทุก epoch ให้ Optuna prune ได้
- `predict_split(run_id, split) -> proba` และ `evaluate(y, proba) -> dict`: ใช้วัดผลทุกที่
- callback `MacroF1Callback`: คำนวณ Val Macro-F1 ทุก epoch, early stopping, เก็บ weights ที่ดีที่สุด
- ข้อมูลป้อนผ่าน `keras.utils.PyDataset` ที่ใช้ `load_pil` + augmentation จาก `fabric_lib.py` (โหลดรูปทีละ batch ไม่โหลดทั้งหมดเข้า RAM)
- Config มี `SMOKE = False`: True = 2 epoch, 2 trial, ข้อมูลบางส่วน (ใช้ทดสอบทั้ง notebook ให้เร็ว ผลลัพธ์เขียนลง `results/_smoke/`, `models/_smoke/`)

## 6. แอป Streamlit (`app.py`)

หน้าแยกด้วย `st.navigation` กราฟใช้ plotly อ่านเฉพาะไฟล์ใน `results/` และ `models/` ไม่ import optuna ไม่คำนวณใหม่

1. **เปรียบเทียบ 4 โมเดล:** การ์ด Top 1, กราฟแท่ง Val Macro-F1 / Accuracy / Precision / Recall / Recall ต่ำสุด, Recall รายคลาส, training curves (เลือกโมเดล), ตารางเวลาเทรน / inference / พารามิเตอร์, เหตุผลที่เลือก
2. **ก่อน vs หลังจูน:** Val และ Test Macro-F1 / Accuracy (mean ± std + จุดราย seed), confusion matrix ก่อน/หลัง, F1 รายคลาส + ส่วนต่าง, ผลแยกแหล่ง, CI + McNemar, Optuna (ค่าราย trial + best-so-far, importance, ตาราง default vs best)
3. **ทำนายรูป:** อัปโหลด/กล้อง → คลาส + ความน่าจะเป็นทั้ง 4 คลาส, เตือนเมื่อความมั่นใจ < 0.6; sidebar ชื่อโมเดล ผล Test ข้อจำกัด (PRD A1–A6); โหลดโมเดลด้วย `st.cache_resource`

ถ้าขั้น 6 ได้โมเดล PyTorch เป็นตัวสุดท้าย `load_predictor` อ่าน `framework` จาก meta แล้วเลือก predictor ตามนั้น (เพิ่มเฉพาะเมื่อเกิดจริง)

## 7. Error handling และการทดสอบ

**Error handling:** แอป: ไฟล์ผลของหน้าใดไม่มี → แสดงว่าต้องรัน notebook ส่วนไหน; รูปเปิดไม่ได้ → ข้อความชัดเจน; ไม่มี GPU → CPU notebook: fingerprint ไม่ตรง → หยุดพร้อมข้อความ; OOM → batch 16; Optuna trial ล้ม → mark fail แล้วไปต่อ, รันซ้ำต่อจาก SQLite

**การทดสอบ (ก่อนรันจริงข้ามคืน):**
1. Spike: ติดตั้ง keras, เทรน 1 epoch ทั้ง 4 โมเดลบน GPU วัดวินาที/epoch, ตรวจว่า `.keras` บันทึก→โหลดแล้วทำนายตรงกัน ถ้าเวลารวมคาดว่าเกิน 6 ชม. ลด trial และแจ้งผู้ใช้
2. Smoke run: `SMOKE=True` + Run All จนจบ ทุกไฟล์ที่แอปใช้ถูกสร้าง
3. ความสอดคล้อง: `KerasPredictor` ให้ความน่าจะเป็นเท่ากับ notebook บน Val 20 รูป (ต่างกัน < 1e-4)
4. แอป: เปิดทั้ง 3 หน้าใน browser และทำนายรูปจริง

## 8. นอกขอบเขต

Grad-CAM, DINOv2/DINOv3 (ViT), SVM/KNN, การเก็บข้อมูลใหม่หรือแก้ label เพิ่ม

## 9. แก้ไข 2026-10-31: เปลี่ยน Keras → PyTorch/torchvision (ผู้ใช้อนุมัติ)

Spike พบว่า Keras 3.15 + torch backend บน RTX 4050 ใช้ไม่ได้จริง (batch 32 เกิน 6 GB, batch 16 หน่วยความจำรั่วทุก step, EfficientNetB0 417 s/epoch, ConvNeXtTiny 523 s/epoch เทียบ PyTorch ~26–50 s/epoch) เวลารวมเกิน 30 ชม. จึงใช้สถาปัตยกรรมเดียวกันจาก torchvision แทน:

| เดิม (Keras) | ใหม่ (torchvision, ImageNet weights) |
|---|---|
| ResNet50V2 | `resnet50` (IMAGENET1K_V2) |
| DenseNet121 | `densenet121` (IMAGENET1K_V1) |
| EfficientNetB0 | `efficientnet_b0` (IMAGENET1K_V1) |
| ConvNeXtTiny | `convnext_tiny` (IMAGENET1K_V1) |

สิ่งที่เปลี่ยนตาม: ทุกโมเดลใช้ normalize ImageNet mean/std (ไม่ต้องมี layer preprocess แยก), mixed precision = `torch.autocast` fp16 + GradScaler, BatchNorm แช่แข็ง = ตั้ง BN เป็น eval + ไม่คำนวณ grad, "layer" สำหรับสัดส่วนการเปิด = โมดูลใบที่มีพารามิเตอร์ของ backbone ตามลำดับ, บันทึกเป็น `.pth` (`models/runs/<run_id>.pth`, `models/best_model.pth`), `model_meta.json` มี `"framework": "torch"` และ `"arch"`. สูตร default, search space, ขั้นตอน, เกณฑ์เลือก และแอปเหมือนเดิมทุกอย่าง ขั้นสำรอง (ข้อ 6) ยังเป็น ConvNeXt V2-Tiny (timm) ตัวเลข spike ของ Keras ใช้อธิบายในรายงานว่าทำไมไม่ใช้ Keras

## 10. แก้ไข 2026-10-31 (ครั้งที่ 2): รอบเทรนตามสไลด์, Unseen จากอินเทอร์เน็ต, กราฟ matplotlib/plotly

ที่มา: สไลด์ "สรุปงานกลุ่มวิชา ML เรื่องการ Classification รูปภาพ" (Durian Group) และคำขอของผู้ใช้ 3 ข้อ

**10.1 Unseen data จากอินเทอร์เน็ต (ใช้วัดผลอย่างเดียว ไม่ใช้เทรน/เลือกโมเดล)**
- แหล่ง: TextileNet fibre subset (Zhong et al., 2023, https://github.com/hahashu/TextileNet) คลาส cotton / polyester / silk / viscose(rayon)
- เป้า: คลาสละ ~100 รูป (รวม ~400) เก็บที่ `unseen_data/<Class>/` + `unseen_data/sources.csv` (file, class, source, original_path/url, verified_by, keep)
- คัดกรองอัตโนมัติ: ตัดรูปซ้ำ (md5 และรูปที่ซ้ำกับ dataset เดิม), ด้านสั้น < 128 px; ทีมตรวจ label และตัดรูปที่ไม่เห็นเนื้อผ้าผ่านหน้ารีวิว ก่อนใช้วัดผล
- ถ้าคลาสใดได้ไม่ถึง 100 รูป ให้รายงานจำนวนจริงและถามผู้ใช้ก่อนหาเพิ่ม
- ข้อจำกัดที่ต้องรายงาน: เป็นรูปสินค้าจากเว็บ (distribution ต่างจากชุดเทรน), label จากคำบรรยายสินค้า, ใช้เพื่อการศึกษาและอ้างอิงที่มา ไม่เผยแพร่รูปต่อ
- ชุด Train/Val/Test เดิม (70/15/15, 4887/326/328) ไม่เปลี่ยน

**10.2 รอบเทรนตามสไลด์ (แทนรอบคัดเลือกเดิม)**: 4 โมเดล × 3 รอบ = 12 run, seed 42, สูตร 2 เฟส default เดิม (lr_head 1e-3, lr_ft 1e-5, เปิด 30%, หัว 3 epoch, fine-tune ≤ 30 epoch, patience 7)

| รอบ | augmentation |
|---|---|
| none | resize ด้านสั้น 256 + center crop 224 (เหมือนตอนวัดผล) |
| v1 | horizontal flip, rotation ±10.8° (3% ของรอบวง), translation 5%, Gaussian noise σ=2%, Gaussian blur p=0.30 |
| v2 | horizontal flip, rotation ±21.6° (6%), translation 8%, brightness ±10%, contrast ±10%, Gaussian noise σ=5%, Gaussian blur p=0.50 |

- Top 1 = (โมเดล, รอบ) ที่ Val Macro-F1 สูงสุด; ต่างไม่เกิน 0.01 → inference เร็วกว่า
- ก่อนจูน = Top 1 ด้วยสูตรของรอบนั้น 3 seed; Optuna 15 trial (search space เดิม แต่ `aug` ∈ {none, v1, v2} และ `mixup` ∈ {False, True}); หลังจูน 3 seed; Test ครั้งเดียว; จากนั้นวัด Unseen
- รายงานผล Test และ Unseen ของทั้ง 12 run หลังเลือกโมเดลแล้ว (ระบุว่าไม่ได้ใช้เลือก)
- ตารางขนาดโมเดล: ขนาดไฟล์ (MB), พารามิเตอร์, ImageNet Top-1/Top-5 (จาก torchvision weights meta), inference ms/รูป
- run เดิมของวันที่ 31 ต.ค. ใช้ซ้ำไม่ได้ (augmentation ต่างกัน) ย้ายไป `backup_2026-10-31/` ก่อนเริ่ม; เวลาโดยประมาณ ~4.5 ชม.

**10.3 กราฟ**
- notebook: matplotlib ทั้งหมด (เลิกใช้ seaborn) บันทึก PNG ใน `results/`
- แอป: plotly ทั้งหมด
- กราฟใหม่ต่อโมเดล×รอบ (ตามสไลด์): (1) Train vs Val vs Test vs Unseen — Accuracy / Precision / Recall / F1 / Loss (train วัดใน eval mode ไม่มี augmentation), (2) P/R/F1 รายคลาสบน Test และ Unseen, (3) learning curve loss/accuracy พร้อมข้อความวินิจฉัย (best epoch, train−val gap, ระดับ overfitting), (4) confusion matrix แบบนับจำนวน

**10.4 แอป**: หน้า 1 ตาราง 12 run + ขนาดโมเดล, หน้า 2 ก่อน/หลังจูน + augmentation, หน้า 3 ทำนายรูป, **หน้า 4 Unseen data** (accuracy/F1 รวมและรายคลาส, ตารางรูปที่ทายผิด, ดูรูปทีละรูปพร้อมผลทำนาย)
