"""สร้าง fabric_classification.ipynb ใหม่ (เวอร์ชันคัดเลือก CNN 4 ตัว -> จูน Top 1 -> แอป)

รัน: .venv\\Scripts\\python.exe tools\\build_notebook.py [--out path.ipynb]
- ส่วน 2–3 (Load/EDA/Clean/Split) คัดลอกจาก notebook เดิม (tools/legacy_notebook_2026-10-30.ipynb) เพื่อให้แบ่งข้อมูลเหมือนเดิมทุกประการ
- fabric_lib.py, tools/tuning_helpers.py และ app_src/app.py ถูกฝังลง notebook จากไฟล์บนดิสก์
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BACKUP_NB = ROOT / "tools" / "legacy_notebook_2026-10-30.ipynb"  # notebook เวอร์ชัน 30 ต.ค. (ตัด output ออก)
REUSED_CELLS = range(10, 19)  # ส่วน 2 Load+EDA และ 3 Clean+Split ของ notebook เดิม
# manifest.csv / split.csv อยู่ที่ results/ เสมอ (ไม่ย้ายไป results/_smoke ตอน SMOKE)
PATH_SUBS = [('RESULTS_DIR / "manifest.csv"', 'SPLIT_DIR / "manifest.csv"'),
             ('RESULTS_DIR / "split.csv"', 'SPLIT_DIR / "split.csv"')]
# notebook ใช้ matplotlib อย่างเดียว: แทน seaborn ในเซลล์ EDA เดิม
PLOT_SUBS = [('sns.scatterplot(data=ok, x="width", y="height", hue="source", s=12, alpha=0.5, ax=axes[0])',
              'for src, g in ok.groupby("source"):\n'
              '    axes[0].scatter(g.width, g.height, s=12, alpha=0.5, label=src)\n'
              'axes[0].legend()'),
             ('sns.histplot(data=ok, x="min_side", hue="source", bins=40, ax=axes[1])',
              'for src, g in ok.groupby("source"):\n'
              '    axes[1].hist(g.min_side, bins=40, alpha=0.6, label=src)\n'
              'axes[1].legend()')]


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def reused_cells():
    old = json.loads(BACKUP_NB.read_text(encoding="utf-8"))["cells"]
    out = []
    for i in REUSED_CELLS:
        src = "".join(old[i]["source"])
        for a, b in PATH_SUBS + PLOT_SUBS:
            src = src.replace(a, b)
        out.append(md(src) if old[i]["cell_type"] == "markdown" else code(src))
    return out


INTRO = r'''
# FabricSort: ระบบจำแนกเศษผ้าเพื่อนำไปรีไซเคิล

**Applied Machine Learning** · จำแนกรูปเศษผ้า 1 รูปเป็น **Cotton / Polyester / Silk / Viscose** พร้อมค่าความมั่นใจ

Notebook นี้ทำงานครบในการกด **Run All** ครั้งเดียว โดยแก้เฉพาะ cell **Config (ส่วนที่ 0)**

**Flow ของเวอร์ชันนี้**
1. เทรน CNN 4 ตัว (ResNet50, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny) ตัวละ **3 รอบ** ตามสไลด์ของวิชา:
   รอบ 1 ไม่ทำ augmentation, รอบ 2 แบบเบา (v1), รอบ 3 แบบหนัก (v2) และรอบ 4 แบบรักษาพื้นผิวผ้า (texture) รวม 16 run
2. เลือก **Top 1 (โมเดล + รอบ)** จาก Val Macro-F1
3. วัดผล **ก่อนจูน** (สูตร default, 3 seed) → จูน hyperparameter ด้วย **Optuna** → วัดผล **หลังจูน** (3 seed)
4. ประเมินบน Test ครั้งเดียว แล้ววัดผลกับ **Unseen data** (รูปผ้าจากอินเทอร์เน็ต) พร้อมกราฟ Train vs Val vs Test vs Unseen
5. export โมเดล และสร้างแอป Streamlit (กราฟใน notebook ใช้ matplotlib ส่วนในแอปใช้ plotly)

> **ทำไมไม่ใช้ Keras:** เริ่มจาก Keras Applications ก่อนตามแผน แต่ Keras 3 (backend PyTorch) บน RTX 4050 ใช้ไม่ได้จริง:
> batch 32 ใช้หน่วยความจำเกิน 6 GB, batch 16 หน่วยความจำรั่วทุก step และช้ามาก (EfficientNetB0 417 วินาที/epoch,
> ConvNeXtTiny 523 วินาที/epoch เทียบกับ PyTorch ~40–50 วินาที/epoch) ส่วน TensorFlow ใช้ GPU บน Windows ไม่ได้
> จึงใช้สถาปัตยกรรมเดียวกันจาก **torchvision** แทน (ResNet50V2 ของ Keras → ResNet50 ของ torchvision)

| ส่วน | เนื้อหา | ผลลัพธ์ |
|---|---|---|
| – | ติดตั้งไลบรารี | แพ็กเกจตาม `requirements.txt` |
| 0 | Config | ตั้งค่าจุดเดียว |
| 1 | Setup + โมดูลที่ใช้ร่วมกับแอป | `fabric_lib.py` |
| 2–3 | Load + EDA, Clean + Split (เหมือนเวอร์ชันก่อน) | `results/manifest.csv`, `results/split.csv` |
| 4 | ฟังก์ชันเทรนและวัดผลร่วม | – |
| 5 | 4 โมเดล × 4 รอบ (none / v1 / v2 / texture) | `results/screening.csv` |
| 6 | เลือก Top 1 | `results/selection.json` |
| 7 | ก่อนจูน (3 seed) | `results/runs.csv` |
| 8 | จูนด้วย Optuna | `results/optuna_trials.csv` |
| 9 | หลังจูน (3 seed) | `results/runs.csv` |
| 10 | Test: ก่อน vs หลัง + Test ของทั้ง 16 run | `results/test_before_after.csv`, `screening_test.csv` ฯลฯ |
| 10b | Generalization (ตามสไลด์) | `results/generalization.csv`, `gen_*.png` |
| 10c | Unseen data | `results/unseen_metrics.csv`, `unseen_predictions.csv` |
| 11 | Error analysis | รูปที่ทายผิด |
| 12 | Export | `models/best_model.pth` + `model_meta.json` |
| 13–14 | เขียนและเปิดแอป | `app.py`, http://localhost:8501 |

**งานวิจัยอ้างอิง**
- **[R1]** Zhu et al. (2023). *Identification of cashmere and wool based on LBP and GLCM texture feature selection.* → baseline SVM (เวอร์ชันแรก)
- **[R2]** Mudaffarsyah et al. (2024). *Klasifikasi Jenis Lamun Menggunakan Ekstraksi Fitur GLCM dan Algoritma KNN.* → baseline KNN (เวอร์ชันแรก)
- **[R3]** Enow Gnoupa et al. (2025). *Multimodal Feature Inputs Enable Improved Automated Textile Identification.* Textiles 5(3). → ResNet50 ดีที่สุดเมื่อเทียบกับ InceptionV3, DenseNet และ VGG-19

เวอร์ชันก่อนหน้า (SVM/KNN และ ResNet50/ConvNeXt V2/DINOv2) ถูกลบออกแล้ว เหลือเฉพาะงานปัจจุบัน
'''

INSTALL_MD = r'''
## ติดตั้งไลบรารี

ตรวจว่าขาดแพ็กเกจอะไร ถ้าครบแล้วจะข้ามทันที เครื่องที่มี GPU NVIDIA จะได้ PyTorch แบบ CUDA ตาม `requirements.txt`
แนะนำให้ใช้ virtual environment ของโปรเจค (ดู README.md)
'''

INSTALL = r'''
# ===== ติดตั้งไลบรารี: ครั้งแรกจะติดตั้ง ครั้งต่อไปจะข้ามเองถ้าครบแล้ว =====
import importlib, importlib.util, subprocess, sys
from pathlib import Path

assert sys.version_info >= (3, 11), f"ต้องใช้ Python 3.11 ขึ้นไป (ตอนนี้ {sys.version.split()[0]})"
REQUIRED = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas",
            "sklearn": "scikit-learn", "scipy": "scipy", "optuna": "optuna", "plotly": "plotly",
            "PIL": "pillow", "joblib": "joblib", "tqdm": "tqdm", "matplotlib": "matplotlib",
            "streamlit": "streamlit"}
missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    if not Path("requirements.txt").exists():
        raise FileNotFoundError("ไม่พบ requirements.txt ให้เปิด notebook จากโฟลเดอร์โปรเจค")
    print("ยังไม่มีแพ็กเกจ:", ", ".join(missing), "-> ติดตั้งตาม requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"])
    importlib.invalidate_caches()
    if any(m in sys.modules for m in ("torch", "numpy")):
        raise SystemExit("ติดตั้งเสร็จแล้ว: ให้ Restart kernel แล้ว Run All อีกครั้ง")
check = subprocess.run([sys.executable, "-c", "import numpy, sklearn.metrics, scipy.stats, torch, torchvision, optuna"],
                       capture_output=True, text=True)
if check.returncode != 0:
    err = check.stderr.strip().splitlines()[-1] if check.stderr.strip() else "unknown error"
    hint = ("\nWindows Smart App Control บล็อกไฟล์ DLL ของไลบรารี ดูหัวข้อ 'ปัญหา Smart App Control' ใน README.md"
            if "Application Control" in err or "WinError 4551" in err else "")
    raise ImportError(f"import ไลบรารีไม่สำเร็จด้วย {sys.executable}:\n{err}{hint}")
print(f"ไลบรารีครบแล้ว (Python {sys.version.split()[0]})")
'''

CONFIG_MD = r'''
## 0. Config

ค่าที่ปรับได้ทั้งหมดอยู่ใน cell นี้ที่เดียว

**โครงสร้างโฟลเดอร์ข้อมูล**
```
dataset/
  iBUG/{Cotton,Polyester,Silk,Viscose}/<ID ชิ้นผ้า>/im_1..im_4.png
  TextileClass7/{Cotton,Polyester,Silk,Viscose}/*.jpg
```

**`SMOKE = True`** ใช้ทดสอบว่า notebook รันครบทุก cell ได้ในไม่กี่นาที (2 epoch, 2 trial, ข้อมูลส่วนน้อย)
ผลลัพธ์จะเขียนแยกไว้ที่ `results/_smoke/` และ `models/_smoke/` ไม่ทับผลจริง
'''

CONFIG = r'''
# ===== Config: แก้เฉพาะ cell นี้ =====
from pathlib import Path

PROJECT_DIR = Path.cwd()                 # โฟลเดอร์ที่มี notebook นี้ (ให้เปิด notebook จากโฟลเดอร์โปรเจค)
DATA_DIR    = PROJECT_DIR / "dataset"    # dataset/<source>/<Class>/...
OUTPUT_DIR  = PROJECT_DIR                # ที่เก็บผลลัพธ์ทั้งหมด

SOURCES = ["iBUG", "TextileClass7"]
# TextileClass7: Roboflow ทำ augmentation ไว้ 3 เวอร์ชันต่อรูป ชื่อไฟล์เป็น "<ต้นฉบับ>_jpg.rf.<hash>.jpg"
# จึงตัด "_jpg.rf.<hash>" ออกเพื่อให้ทุกเวอร์ชันของรูปเดียวกันเป็นชิ้นผ้าเดียวกัน
GROUP_REGEX = {"iBUG": None, "TextileClass7": r"_(jpe?g|png)\.rf\.[0-9a-f]+$"}
FABRIC_GROUPS_CSV = PROJECT_DIR / "data_audit" / "fabric_groups.csv"  # รูปหลายรูปของผ้าชิ้นเดียวกัน
EXCLUDE_CSV       = PROJECT_DIR / "data_audit" / "exclude.csv"        # รูปที่ label ขัดแย้ง
EVAL_ONE_PER_PIECE = True                # Val/Test ใช้ 1 รูปต่อชิ้นผ้า

SEED     = 42
IMG_SIZE = 224
MIN_SIDE = 64
SPLIT    = (0.70, 0.15, 0.15)            # Train / Val / Test
N_JOBS   = -1                            # จำนวน process ตอนสแกนรูป

# ===== โมเดลและการเทรน =====
MODEL_KEYS  = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"]   # torchvision
ROUNDS = ["none", "v1", "v2", "texture"] # รอบตามสไลด์ (none/v1/v2) + texture = augmentation ที่รักษาพื้นผิวผ้า
UNSEEN_DIR = PROJECT_DIR / "unseen_data" # รูปจากอินเทอร์เน็ต (TextileNet) ใช้วัดผลอย่างเดียว
SEEDS       = (42, 7, 2024)              # seed ของผล "ก่อนจูน" และ "หลังจูน"
MAX_EPOCHS  = 30                         # epoch สูงสุดของเฟส fine-tune
PATIENCE    = 7                          # early stopping บน Val Macro-F1
BATCH_SIZE  = 32                         # GPU out-of-memory -> ลดเป็น 16 ให้อัตโนมัติ
NUM_WORKERS = 4
GPU_TEMP_PAUSE  = None                   # None = ไม่พักอัตโนมัติ (ผู้ใช้ดูอุณหภูมิและสั่งหยุดเอง); ใส่ตัวเลข เช่น 60 เพื่อพักเมื่อถึง °C นี้
GPU_TEMP_RESUME = 50                     # ถ้าเปิดการพักอัตโนมัติ: พักจนเย็นลงถึงกี่ °C แล้วจึงเทรนต่อ
N_TRIALS     = 15                        # จำนวน trial ของ Optuna
TRIAL_EPOCHS = 12                        # epoch ของเฟส fine-tune ใน 1 trial (สั้นกว่าเทรนจริงเพื่อประหยัดเวลา)

F1_TARGET      = 0.85                    # เป้า Test Macro-F1 ตาม PRD (ไม่ถึง -> ต้องทำขั้นสำรอง)
CONF_THRESHOLD = 0.60                    # ความมั่นใจต่ำกว่านี้ แอปจะเตือนให้คนตรวจซ้ำ
SKIP_TRAINING_IF_EXISTS = True           # มีผลของ run เดียวกันอยู่แล้ว -> โหลดมาใช้แทนการเทรนใหม่
LAUNCH_APP = True
APP_PORT   = 8501

SMOKE = False                            # True = ทดสอบเร็ว (ผลไม่ใช้จริง เขียนแยกที่ *_smoke/)
SMOKE_PER_CLASS = 48
if SMOKE:
    MAX_EPOCHS, TRIAL_EPOCHS, N_TRIALS, SEEDS, PATIENCE = 2, 2, 2, (42, 7), 2

# โฟลเดอร์ผลลัพธ์ (ไม่ต้องแก้)
SPLIT_DIR   = OUTPUT_DIR / "results"                               # manifest.csv, split.csv (ใช้ร่วมกันเสมอ)
RESULTS_DIR = OUTPUT_DIR / "results" / ("_smoke" if SMOKE else "")
MODELS_DIR  = OUTPUT_DIR / "models" / ("_smoke" if SMOKE else "")
RUNS_DIR    = MODELS_DIR / "runs"                                  # <run_id>.pth
HIST_DIR    = RESULTS_DIR / "histories"                            # <run_id>.json
for d in (SPLIT_DIR, RESULTS_DIR, MODELS_DIR, RUNS_DIR, HIST_DIR):
    d.mkdir(parents=True, exist_ok=True)
'''

SETUP_MD = r'''
## 1. Setup

### 1.1 โมดูลที่ใช้ร่วมกับแอป (`fabric_lib.py`)

โหลดรูป (หมุนตาม EXIF), augmentation, Dataset, โมเดล 4 ตัวจาก torchvision (เปลี่ยนชั้นสุดท้ายเป็น Dropout + Linear 4 คลาส),
ฟังก์ชันเปิด/แช่แข็ง layer และ predictor ที่แอปใช้ ทั้ง notebook และแอปใช้โค้ดชุดเดียวกัน preprocessing จึงตรงกันเสมอ
'''

IMPORTS_MD = r'''
### 1.2 Import, fix seed และตรวจ GPU

`evaluate` วัดผลทุกโมเดลด้วยเกณฑ์เดียวกัน (Accuracy, Precision/Recall/F1 แบบ macro และรายคลาส, Recall ต่ำสุด)
'''

IMPORTS = r'''
import sys, os, json, time, random, re, hashlib, warnings, platform, importlib, io, math, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import sklearn, PIL
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch, torchvision, optuna
import torch.nn.functional as F
from IPython.display import display, HTML

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))  # ให้ process ลูกของ DataLoader import fabric_lib ได้
import fabric_lib as fl
importlib.reload(fl)

CLASSES = fl.CLASSES
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def evaluate(y_true, proba):
    """วัดผลจาก probability (n, 4) -> dict ของตัวชี้วัด (คลาสที่ไม่มีตัวอย่างในชุดนั้นได้ NaN)"""
    y_pred = np.asarray(proba).argmax(1)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    has = support > 0
    p, r, f = (np.where(has, v, np.nan) for v in (p, r, f))
    m = {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": np.nanmean(p),
         "macro_recall": np.nanmean(r), "macro_f1": np.nanmean(f), "min_recall": np.nanmin(r)}
    for i, c in enumerate(CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = p[i], r[i], f[i]
    return {k: float(v) for k, v in m.items()}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")


seed_everything()
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3,
                     "font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"]})  # แสดงภาษาไทยในกราฟได้
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore", message=".*least populated class.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "optuna": optuna.__version__}
display(pd.Series(versions, name="version").to_frame())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 2**30:.1f} GB), CUDA {torch.version.cuda}")
else:
    print("ไม่พบ GPU ที่ใช้ CUDA ได้: จะเทรนบน CPU ซึ่งช้ามาก ตรวจการติดตั้ง PyTorch แบบ CUDA ใน README.md")
'''

SPLIT_CHECK = r'''
# ตรวจข้อมูลก่อนเทรน: จำนวนรูปต้องเท่ากับที่ใช้ในเวอร์ชันก่อน เพื่อให้ผลเทียบกันได้
if SMOKE:
    rng_s = np.random.default_rng(SEED)
    keep = np.zeros(len(split_df), dtype=bool)
    for (s, c), idx in split_df.groupby(["split", "class"]).indices.items():
        n = SMOKE_PER_CLASS if s == "train" else SMOKE_PER_CLASS // 3
        keep[rng_s.choice(idx, min(n, len(idx)), replace=False)] = True
    split_df = split_df[keep].reset_index(drop=True)
    y_all = split_df["class"].map(CLASS_TO_IDX).to_numpy()
    masks = {s: (split_df["split"] == s).to_numpy() for s in ("train", "val", "test")}
    print("SMOKE: ใช้ข้อมูลส่วนน้อย", {s: int(m.sum()) for s, m in masks.items()})

y_tr, y_va, y_te = (y_all[masks[s]] for s in ("train", "val", "test"))
N_IMAGES = tuple(int(masks[s].sum()) for s in ("train", "val", "test"))
if not SMOKE and N_IMAGES != (4887, 326, 328):
    raise RuntimeError(f"จำนวนรูป Train/Val/Test = {N_IMAGES} ไม่ตรงกับเวอร์ชันก่อน (4887, 326, 328): "
                       "ข้อมูลหรือ split.csv เปลี่ยนไป ผลจะเทียบกับเวอร์ชันก่อนไม่ได้")

# fingerprint ของข้อมูล: run ที่บันทึกไว้จะถูกใช้ซ้ำเฉพาะเมื่อข้อมูลชุดเดียวกัน
FINGERPRINT = hashlib.md5(json.dumps({"split": split_df[["path", "split"]].values.tolist(),
                                      "seed": SEED}).encode()).hexdigest()[:12]
print(f"train {N_IMAGES[0]:,} | val {N_IMAGES[1]:,} | test {N_IMAGES[2]:,} | data fingerprint {FINGERPRINT}")
'''

TRAIN_MD = r'''
## 4. ฟังก์ชันเทรนและวัดผลร่วม

ทุกขั้น (คัดเลือก, ก่อนจูน, trial ของ Optuna, หลังจูน) ใช้ฟังก์ชันเทรนเดียวกัน `train_one(arch, params, seed, max_epochs, stage)`

**สูตรเทรนแบบ 2 เฟส (transfer learning มาตรฐาน)**
1. **เฟส 1 (head):** แช่แข็ง backbone ที่ pretrain บน ImageNet แล้วเทรนเฉพาะหัวจำแนก `head_epochs` epoch ด้วย lr `lr_head`
2. **เฟส 2 (fine-tune):** เปิดให้เทรน layer บนสุด `unfreeze_frac` ของ backbone ด้วย lr `lr_ft` ที่ลดแบบ cosine
   สูงสุด `MAX_EPOCHS` epoch และหยุดเมื่อ Val Macro-F1 ไม่ดีขึ้น `PATIENCE` epoch แล้วใช้ weights ของ epoch ที่ดีที่สุด

BatchNorm แช่แข็งเสมอ (ใช้ค่าสถิติจาก ImageNet) loss เป็น cross-entropy ถ่วงน้ำหนักตามสัดส่วนคลาส ใช้ mixed precision (AMP)
และ AdamW ทุก run ถูกบันทึกด้วย `run_id` (hash ของโมเดล + hyperparameter + seed + epoch + ข้อมูล) รันซ้ำจะโหลดของเดิม

**สูตร default** (`DEFAULT_PARAMS`): lr_head 1e-3, lr_ft 1e-5, เปิด 30% บนสุด, เฟส 1 3 epoch, dropout 0.3,
weight decay 1e-4, ไม่มี label smoothing, ไม่มี MixUp/CutMix และ augmentation ตามรอบ (none / v1 / v2)
'''

TRAIN = r'''
RUNS_CSV = RESULTS_DIR / "runs.csv"
# เพิ่มเลขนี้เมื่อแก้โค้ดเทรน (_fit / fabric_lib) run ที่ cache จากเวอร์ชันก่อนจะมีคำเตือน
# (ถ้าต้องการเทรนใหม่ทั้งหมด ให้ลบ models/runs/ และ results/histories/)
CODE_VERSION = 3
counts = np.bincount(y_tr, minlength=len(CLASSES))
CLASS_W = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
print("class weights:", dict(zip(CLASSES, CLASS_W.round(2))))
_PROBA_CACHE = {}


def make_loader(split, train, batch_size, aug="none", seed=SEED):
    m = masks[split]
    ds = fl.FabricDataset([DATA_DIR / p for p in split_df["path"][m]], y_all[m], train=train, aug=aug)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                                       num_workers=NUM_WORKERS, pin_memory=USE_AMP,
                                       persistent_workers=NUM_WORKERS > 0,
                                       generator=torch.Generator().manual_seed(seed))


COOL_LOG = {"pauses": 0, "seconds": 0}


def cool_gpu():
    """ถ้าการ์ดจอถึง GPU_TEMP_PAUSE °C ให้หยุดรอจนเย็นถึง GPU_TEMP_RESUME °C (กันการ์ดจอร้อนสะสมจนไดรเวอร์หลุด)"""
    if not USE_AMP or GPU_TEMP_PAUSE is None:
        return
    waited = wait_for_cool(torch.cuda.temperature, GPU_TEMP_PAUSE, GPU_TEMP_RESUME)
    if waited:
        COOL_LOG["pauses"] += 1
        COOL_LOG["seconds"] += waited
        print(f"  พักการ์ดจอ {waited:.0f} วินาที (ถึง {GPU_TEMP_PAUSE}°C) รวม {COOL_LOG['pauses']} ครั้ง "
              f"{COOL_LOG['seconds'] / 60:.0f} นาที")


@torch.inference_mode()
def predict_model(model, loader):
    model.eval()
    out = []
    for i, (x, _) in enumerate(loader):
        if i % 10 == 0:
            cool_gpu()
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(x.to(DEVICE, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def ce_loss(y, proba):
    """cross-entropy เฉลี่ย (ไม่ถ่วงน้ำหนัก) จาก probability"""
    return float(-np.log(np.clip(np.asarray(proba)[np.arange(len(y)), y], 1e-7, 1)).mean())


def run_id_of(arch, params, seed, max_epochs, trial=False):
    key = {"arch": arch, "params": params, "seed": seed, "max_epochs": max_epochs, "data": FINGERPRINT}
    if trial:  # trial ไม่เก็บ weights จึงแยก id จาก run เต็มที่ใช้ค่าเดียวกัน
        key["trial"] = True
    return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _fit(arch, params, seed, max_epochs, batch_size, trial=None):
    """เทรน 2 เฟส -> (state_dict ที่ดีที่สุด, history, val metrics ของ epoch ที่ดีที่สุด, วินาที, จำนวนพารามิเตอร์)"""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = fl.build_model(arch, dropout=params["dropout"], pretrained=True).to(DEVICE)
    train_loader = make_loader("train", True, batch_size, params["aug"], seed)
    val_loader = make_loader("val", False, 64, seed=seed)
    cw = torch.tensor(CLASS_W, dtype=torch.float32, device=DEVICE)
    ls = float(params["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history, best_f1, best_state, best_vm, epoch = [], -1.0, None, None, 0
    t0 = time.perf_counter()
    phases = [("head", int(params["head_epochs"]), params["lr_head"], 0.0),
              ("finetune", int(max_epochs), params["lr_ft"], params["unfreeze_frac"])]
    for phase, n_epochs, lr, frac in phases:
        fl.set_trainable(model, frac)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                weight_decay=params["weight_decay"])
        total_steps, step, wait = max(1, n_epochs * len(train_loader)), 0, 0
        for e in range(n_epochs):
            epoch += 1
            fl.train_mode(model)
            tot, n, correct = 0.0, 0, 0
            for x, y in train_loader:
                if step % 10 == 0:
                    cool_gpu()
                x = x.to(DEVICE, non_blocking=True)
                target = F.one_hot(y.to(DEVICE), len(CLASSES)).float()
                if params["mixup"]:
                    x, target = fl.mix_batch(x, target, rng)
                if phase == "finetune":  # cosine decay ภายในเฟส fine-tune
                    opt.param_groups[0]["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
                step += 1
                soft = target * (1 - ls) + ls / len(CLASSES)
                with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                    logits = model(x)
                w = target @ cw
                loss = (-(soft * F.log_softmax(logits.float(), dim=1)).sum(1) * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(x)
                correct += (logits.argmax(1) == y.to(DEVICE)).sum().item()
                n += len(x)
            vp = predict_model(model, val_loader)
            vm = evaluate(y_va, vp)
            history.append({"epoch": epoch, "phase": phase, "train_loss": tot / max(n, 1),
                            "train_accuracy": correct / max(n, 1), "val_loss": ce_loss(y_va, vp),
                            "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
                            "lr": opt.param_groups[0]["lr"]})
            improved = vm["macro_f1"] > best_f1 + 1e-4
            print(f"  epoch {epoch:2d} [{phase:8s}] loss {tot / max(n, 1):.3f} | val acc {vm['accuracy']:.3f} "
                  f"F1 {vm['macro_f1']:.3f}{'  *' if improved else ''} | {time.perf_counter() - t0:.0f}s")
            if improved:
                best_f1, best_vm, wait = vm["macro_f1"], vm, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif phase == "finetune":
                wait += 1
            if trial is not None and phase == "finetune":
                trial.report(vm["macro_f1"], e)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if phase == "finetune" and wait >= PATIENCE:
                print(f"  early stopping (Val Macro-F1 ไม่ดีขึ้น {PATIENCE} epoch)")
                break
    n_params = sum(p.numel() for p in model.parameters())
    return best_state, history, best_vm, time.perf_counter() - t0, n_params


def load_run(run_id):
    return json.loads((HIST_DIR / f"{run_id}.json").read_text(encoding="utf-8"))


def train_one(arch, params, seed, max_epochs, stage, trial=None):
    """เทรน (หรือโหลดถ้ามีแล้ว) 1 run -> run_id; trial ของ Optuna เก็บเฉพาะ history ไม่เก็บ weights"""
    rid = run_id_of(arch, params, seed, max_epochs, trial=stage == "trial")
    hist_path, w_path = HIST_DIR / f"{rid}.json", RUNS_DIR / f"{rid}.pth"
    if SKIP_TRAINING_IF_EXISTS and hist_path.exists() and (stage == "trial" or w_path.exists()):
        r = load_run(rid)
        print(f"โหลด {arch} seed {seed} ({stage}) จาก run {rid}: Val Macro-F1 {r['val_metrics']['macro_f1']:.3f}")
        if r.get("code_version") != CODE_VERSION:
            warnings.warn(f"run {rid} เทรนด้วยโค้ดเวอร์ชัน {r.get('code_version')} แต่ตอนนี้เป็น {CODE_VERSION}: "
                          "ถ้าแก้โค้ดเทรน ให้ลบ models/runs/ และ results/histories/ แล้วรันใหม่")
        return rid
    print(f"เทรน {fl.TV_ZOO[arch]['name']} | {stage} | seed {seed} | run {rid}")
    for bs in dict.fromkeys((BATCH_SIZE, 16)):
        try:
            state, history, vm, secs, n_params = _fit(arch, params, seed, max_epochs, bs, trial)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"GPU out-of-memory ที่ batch {bs} ลอง batch ที่เล็กลง")
    else:
        raise RuntimeError("GPU memory ไม่พอแม้ใช้ batch 16")
    if stage != "trial":
        torch.save(state, w_path)
    record = {"run_id": rid, "arch": arch, "stage": stage, "seed": seed, "max_epochs": max_epochs, "params": params,
              "batch_size": bs, "history": history, "val_metrics": vm, "train_time_s": secs, "n_params": n_params,
              "epochs_trained": len(history), "data_fingerprint": FINGERPRINT, "code_version": CODE_VERSION}
    hist_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    row = {"run_id": rid, "stage": stage, "arch": arch, "seed": seed, "max_epochs": max_epochs, "batch_size": bs,
           **{f"p_{k}": v for k, v in params.items()}, **{f"val_{k}": v for k, v in vm.items()},
           "train_time_s": round(secs, 1), "epochs_trained": len(history)}
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)
    print(f"-> Val Macro-F1 {vm['macro_f1']:.3f} | Acc {vm['accuracy']:.3f} | {secs / 60:.1f} นาที")
    torch.cuda.empty_cache()
    return rid


def load_model(run_id):
    r = load_run(run_id)
    model = fl.build_model(r["arch"], dropout=r["params"]["dropout"], pretrained=False)
    model.load_state_dict(torch.load(RUNS_DIR / f"{run_id}.pth", map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def predict_split(run_id, split):
    """probability (n, 4) ของ run บนชุด val/test (cache ไว้ในหน่วยความจำ)"""
    if (run_id, split) not in _PROBA_CACHE:
        _PROBA_CACHE[(run_id, split)] = predict_model(load_model(run_id), make_loader(split, False, 64))
    return _PROBA_CACHE[(run_id, split)]


def time_inference(run_id, n=50):
    """เวลา end-to-end ต่อ 1 รูป (อ่านไฟล์ + preprocessing + ทำนาย) ด้วย predictor ตัวเดียวกับแอป, ค่ามัธยฐาน"""
    pred = fl.TorchPredictor(load_model(run_id), DEVICE)
    sample = [DATA_DIR / p for p in split_df["path"][masks["val"]].iloc[:n]]
    pred.predict_proba(sample[:2])
    times = []
    for p in sample:
        t0 = time.perf_counter()
        pred.predict_proba([p])
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def plot_history(run_ids, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for label, rid in run_ids.items():
        h = pd.DataFrame(load_run(rid)["history"])
        axes[0].plot(h.epoch, h.train_loss, marker=".", label=label)
        axes[1].plot(h.epoch, h.val_macro_f1, marker=".", label=label)
    axes[0].set(title=f"{title}: train loss", xlabel="epoch")
    axes[1].set(title=f"{title}: Val Macro-F1", xlabel="epoch")
    axes[1].legend()
    savefig(fig, fname)
    plt.show()


def plot_cm(ax, cm, counts_only=False):
    """confusion matrix ด้วย matplotlib: สีตามสัดส่วนของคลาสจริง ตัวเลขเป็นจำนวน (และ %)"""
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}" if counts_only else f"{cm[i, j]}\n{norm[i, j]:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="predicted", ylabel="true")
    ax.grid(False)
'''

SCREEN_MD = r'''
## 5. รอบเทรน: 4 โมเดล × 4 รอบ (สูตร default, seed 42)

| # | โมเดล | ประเภท | พารามิเตอร์ | ปี | ทำไมเลือก |
|---|---|---|---|---|---|
| 1 | **ResNet50** | CNN แบบ residual | ~25M | 2015 | baseline ตาม PRD และ R3 (ResNet50 ดีที่สุดในงานจำแนกผ้า) |
| 2 | **DenseNet121** | CNN แบบ dense connection | ~8M | 2017 | R3 ใช้เปรียบเทียบ, ใช้ฟีเจอร์ระดับต่ำซ้ำได้ดี (เหมาะกับพื้นผิว) |
| 3 | **EfficientNet-B0** | CNN ที่ปรับขนาดอย่างสมดุล | ~5M | 2019 | เล็กและเร็ว |
| 4 | **ConvNeXt-Tiny** | CNN ยุคใหม่ (ออกแบบตาม transformer) | ~28M | 2022 | CNN ที่แรงที่สุดที่ยังพอดีกับ GPU 6 GB |

**3 รอบต่อโมเดล (ตามสไลด์ตัวอย่างของวิชา)** เปลี่ยนเฉพาะ augmentation ของชุด Train ส่วนอื่นเหมือนกันทุกรอบ

| รอบ | augmentation |
|---|---|
| 1. none | ไม่ทำอะไร (ย่อรูป + ครอปกลาง เหมือนตอนวัดผล) |
| 2. v1 (เบา) | กลับด้านแนวนอน, หมุน ±10.8°, เลื่อน 5%, noise 2%, blur 30% |
| 3. v2 (หนัก) | กลับด้านแนวนอน, หมุน ±21.6°, เลื่อน 8%, ความสว่าง/คอนทราสต์ ±10%, noise 5%, blur 50% |
| 4. texture | **เพิ่มเติม:** สุ่มครอป 40–100%, หมุนทีละ 90°, กลับด้าน, ปรับสีเล็กน้อย (ไม่มี blur / noise / หมุนเอียง) |

รอบ 4 เพิ่มเข้ามาหลังพบว่า v1/v2 ทำให้ทุกโมเดลแย่ลง: blur และ noise ลบรายละเอียดลายเส้นใยที่ใช้แยกชนิดผ้า
และการหมุนเอียงทำให้เกิดขอบดำ ส่วน texture เป็น augmentation ที่ใช้ในเวอร์ชันก่อน (ResNet50 ได้ Val 0.807)

ตารางขนาดโมเดลมีขนาดไฟล์ (MB), จำนวนพารามิเตอร์ และความแม่นยำ ImageNet Top-1/Top-5 ของ weights ที่ใช้ (จาก torchvision)
'''

SCREEN = r'''
SCREEN_RUNS = {(arch, r): train_one(arch, {**DEFAULT_PARAMS, "aug": r}, SEED, MAX_EPOCHS, "screening")
               for arch in MODEL_KEYS for r in ROUNDS}

rows = []
for (arch, r), rid in SCREEN_RUNS.items():
    run, vm = load_run(rid), load_run(rid)["val_metrics"]
    top1, top5 = fl.imagenet_metrics(arch)
    rows.append({"model": fl.TV_ZOO[arch]["name"], "arch": arch, "round": r, "candidate": f"{arch}|{r}", "run_id": rid,
                 **{f"val_{k}": vm[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "min_recall")},
                 **{f"recall_{c}": vm[f"recall_{c}"] for c in CLASSES},
                 "params_M": run["n_params"] / 1e6, "size_mb": (RUNS_DIR / f"{rid}.pth").stat().st_size / 2**20,
                 "imagenet_top1": top1, "imagenet_top5": top5,
                 "train_time_s": run["train_time_s"], "epochs_trained": run["epochs_trained"],
                 "train_accuracy_eval": (tr_acc := evaluate(y_tr, predict_split(rid, "train"))["accuracy"]),
                 "fit": diagnose_fit(run["history"], tr_acc), "infer_ms_per_img": time_inference(rid)})
screening = pd.DataFrame(rows)
screening.to_csv(RESULTS_DIR / "screening.csv", index=False)
display(screening.pivot(index="model", columns="round", values="val_macro_f1")[ROUNDS].round(3))
display(screening.drop_duplicates("model")[["model", "size_mb", "params_M", "imagenet_top1", "imagenet_top5", "infer_ms_per_img"]]
        .set_index("model").round(2))

fig, ax = plt.subplots(figsize=(10, 4))
piv = screening.pivot(index="model", columns="round", values="val_macro_f1")[ROUNDS]
for k, r in enumerate(ROUNDS):
    bars = ax.bar(np.arange(len(piv)) + (k - 1) * 0.27, piv[r], width=0.27, label=r)
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
ax.set(xticks=np.arange(len(piv)), xticklabels=piv.index, ylim=(0, 1), ylabel="Val Macro-F1",
       title="รอบคัดเลือก: Val Macro-F1 (โมเดล × รอบ augmentation)")
ax.legend(title="round")
savefig(fig, "screening_rounds.png")
plt.show()
'''

HELPERS_MD = r'''
## 6. เลือก Top 1 (โมเดล + รอบ)

**เกณฑ์ (กำหนดไว้ล่วงหน้าใน PRD):** Val Macro-F1 สูงสุดจากทั้ง 16 run ถ้าตัวอื่นต่างไม่เกิน 0.01 เลือกตัวที่ inference เร็วกว่า
ชุด Test และ Unseen ยังไม่ถูกใช้ในขั้นนี้

ฟังก์ชันช่วยของส่วน 6–10 (search space ของ Optuna, เกณฑ์เลือก, วินิจฉัย overfitting, bootstrap CI, McNemar) อยู่ใน cell ด้านล่าง
'''

SELECT = r'''
TOP1_CANDIDATE = select_top1(screening)
TOP1, TOP1_ROUND = TOP1_CANDIDATE.split("|")
TOP1_PARAMS = {**DEFAULT_PARAMS, "aug": TOP1_ROUND}
TOP1_NAME = fl.TV_ZOO[TOP1]["name"]
best_f1 = screening.val_macro_f1.max()
selection = {"top1_arch": TOP1, "top1_round": TOP1_ROUND, "top1_name": TOP1_NAME, "top1_candidate": TOP1_CANDIDATE,
             "rule": "max Val Macro-F1 over model x round; within 0.01 -> faster inference (ms/img)",
             "candidates_within_0.01": screening.loc[screening.val_macro_f1 >= best_f1 - 0.01, "candidate"].tolist(),
             "val_macro_f1": dict(zip(screening.candidate, screening.val_macro_f1)),
             "infer_ms_per_img": dict(zip(screening.candidate, screening.infer_ms_per_img))}
(RESULTS_DIR / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
display(screening.sort_values("val_macro_f1", ascending=False)[["model", "round", "val_macro_f1", "val_accuracy",
                                                                 "fit", "infer_ms_per_img"]].round(3))
print(f"Top 1 = {TOP1_NAME} รอบ {TOP1_ROUND} (กลุ่มที่ต่างไม่เกิน 0.01: {selection['candidates_within_0.01']})")
'''

BEFORE_MD = r'''
## 7. ก่อนจูน: Top 1 สูตรของรอบที่ชนะ 3 seed

ผลของ seed เดียวแปรปรวนราว ±0.01–0.02 จึงเทรนสูตรเดิมของ Top 1 ด้วย 3 seed แล้วรายงาน mean ± std
seed 42 เป็น run เดียวกับรอบคัดเลือก (โหลดมาใช้ ไม่เทรนซ้ำ)
'''

BEFORE = r'''
BEFORE_RUNS = [train_one(TOP1, TOP1_PARAMS, s, MAX_EPOCHS, "before") for s in SEEDS]
before_val = pd.DataFrame([{"seed": s, **load_run(r)["val_metrics"]} for s, r in zip(SEEDS, BEFORE_RUNS)]).set_index("seed")
display(before_val[["accuracy", "macro_f1", "min_recall"]].round(3))
print(f"ก่อนจูน Val Macro-F1 = {before_val.macro_f1.mean():.3f} ± {before_val.macro_f1.std(ddof=0):.3f}")
'''

OPTUNA_MD = r'''
## 8. จูน hyperparameter ด้วย Optuna

- **Sampler:** TPE (seed 42) เลือกค่าถัดไปจากผลของ trial ก่อนหน้า
- **Pruner:** MedianPruner หยุด trial ที่ Val Macro-F1 แย่กว่าค่ามัธยฐานของ trial ก่อนหน้าตั้งแต่ epoch ที่ 4 ของเฟส fine-tune
- **Objective:** Val Macro-F1 ของ epoch ที่ดีที่สุด (seed 42, เฟส fine-tune `TRIAL_EPOCHS` epoch)
- ผลทุก trial เก็บใน `results/optuna.db` ถ้า notebook หยุดกลางทาง รันใหม่จะทำต่อจาก trial ที่ค้าง จนครบ `N_TRIALS`

| hyperparameter | ช่วงที่ค้น |
|---|---|
| lr_head | 1e-4 – 3e-3 (log) |
| lr_ft | 1e-6 – 1e-4 (log) |
| unfreeze_frac | 0.1 – 1.0 |
| head_epochs | 1 – 5 |
| dropout | 0 – 0.5 |
| weight_decay | 1e-5 – 0.05 (log) |
| label_smoothing | 0 – 0.15 |
| aug | none, v1, v2, texture |
| mixup (MixUp/CutMix) | ไม่ใช้, ใช้ |
'''

OPTUNA = r'''
storage = f"sqlite:///{(RESULTS_DIR / 'optuna.db').as_posix()}"
# ss2 = search space ที่มี texture (ชื่อ study ใหม่ เพราะตัวเลือกของ aug เปลี่ยน)
study = optuna.create_study(study_name=f"top1_{TOP1}_{TOP1_ROUND}_ss2", storage=storage, direction="maximize",
                            load_if_exists=True,
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3))
assert study.direction == optuna.study.StudyDirection.MAXIMIZE and (RESULTS_DIR / "optuna.db").exists(), \
    "study ต้อง maximize Val Macro-F1 และบันทึกลง results/optuna.db"
study.sampler = make_sampler(len(study.trials), SEED)  # รันต่อแล้วไม่สุ่มค่าซ้ำกับ trial เดิม


def objective(trial):
    params = suggest_params(trial)
    try:
        rid = train_one(TOP1, params, SEED, TRIAL_EPOCHS, "trial", trial=trial)
    finally:  # trial ที่ถูก prune/ล้ม ต้องคืนหน่วยความจำ GPU ก่อน trial ถัดไป
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    trial.set_user_attr("run_id", rid)
    return load_run(rid)["val_metrics"]["macro_f1"]


n_stale = fail_stale_trials(study)  # trial ที่ค้างจากการหยุดกลางทาง
if n_stale:
    print(f"ปิด trial ที่ค้างอยู่ {n_stale} trial เป็น FAIL")
n_left = remaining_trials(study, N_TRIALS)
print(f"Optuna: มี {len(study.trials)} trial แล้ว จะรันเพิ่ม {n_left} trial")
if n_left:
    study.optimize(objective, n_trials=n_left, catch=(Exception,), gc_after_trial=True)

trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration", "user_attrs"))
trials["duration_min"] = trials.pop("duration").dt.total_seconds() / 60
trials.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not complete:
    raise RuntimeError("ไม่มี trial ที่สำเร็จเลย ดูข้อความ error ด้านบน")
try:
    imp = optuna.importance.get_param_importances(study)
except Exception as e:  # trial สำเร็จน้อยเกินไป
    print("คำนวณ importance ไม่ได้:", e)
    imp = {}
pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(RESULTS_DIR / "optuna_importance.csv", index=False)

BEST_PARAMS = {**TOP1_PARAMS, **study.best_params}
print(f"trial ที่ดีที่สุด #{study.best_trial.number}: Val Macro-F1 {study.best_value:.3f}")
display(pd.DataFrame({"ก่อนจูน": TOP1_PARAMS, "หลังจูน": BEST_PARAMS}))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
done = trials[trials.state == "COMPLETE"]
axes[0].scatter(done.number, done.value, label="trial (complete)")
axes[0].plot(done.number, done.value.cummax(), color="C1", label="best so far")
pruned = trials[trials.state == "PRUNED"]
axes[0].scatter(pruned.number, [done.value.min()] * len(pruned), marker="x", color="gray", label="pruned")
axes[0].set(title="Optuna: Val Macro-F1 per trial", xlabel="trial")
axes[0].legend()
if imp:
    pd.Series(imp).sort_values().plot.barh(ax=axes[1], title="Hyperparameter importance")
savefig(fig, "optuna_history.png")
plt.show()
'''

AFTER_MD = r'''
## 9. หลังจูน: Top 1 สูตรที่ดีที่สุด 3 seed

ใช้ hyperparameter จาก trial ที่ดีที่สุด เทรนเต็ม `MAX_EPOCHS` epoch ด้วย seed ชุดเดียวกับ "ก่อนจูน"
'''

AFTER = r'''
AFTER_RUNS = [train_one(TOP1, BEST_PARAMS, s, MAX_EPOCHS, "after") for s in SEEDS]
after_val = pd.DataFrame([{"seed": s, **load_run(r)["val_metrics"]} for s, r in zip(SEEDS, AFTER_RUNS)]).set_index("seed")
cmp = pd.DataFrame({v: {"Val Macro-F1": f"{d.macro_f1.mean():.3f} ± {d.macro_f1.std(ddof=0):.3f}",
                        "Val Accuracy": f"{d.accuracy.mean():.3f} ± {d.accuracy.std(ddof=0):.3f}"}
                    for v, d in (("ก่อนจูน", before_val), ("หลังจูน", after_val))})
display(cmp)
plot_history({f"before s{s}": r for s, r in zip(SEEDS, BEFORE_RUNS)} | {f"after s{s}": r for s, r in zip(SEEDS, AFTER_RUNS)},
             f"{TOP1_NAME}", "before_after_curves.png")
'''

TEST_MD = r'''
## 10. ประเมินบนชุด Test: ก่อน vs หลังจูน (ครั้งเดียว)

ชุด Test ถูกใช้ครั้งแรกที่นี่ หลังจากเลือกโมเดลและจูนเสร็จแล้ว รายงาน
- Macro-F1 / Accuracy แบบ mean ± std ของ 3 seed
- รายคลาส, แยกแหล่งข้อมูล, confusion matrix (ใช้ seed ที่ Val Macro-F1 สูงสุดเป็นตัวแทนของแต่ละฝั่ง)
- ช่วงความเชื่อมั่น 95% (bootstrap 1,000 รอบ) และ McNemar test ระหว่างก่อนกับหลังจูน (p < 0.05 = ต่างกันจริง)

ถ้า Test Macro-F1 เฉลี่ยของ "หลังจูน" ต่ำกว่า `F1_TARGET` (0.85) ต้องทำ**ขั้นสำรอง**: เทรน ConvNeXt V2-Tiny (timm) ด้วย flow เดียวกัน
(การตัดสินใจนี้ดูผล Test จึงบันทึกไว้ใน `selection.json` อย่างชัดเจน)
'''

TEST = r'''
VARIANTS = {"before": BEFORE_RUNS, "after": AFTER_RUNS}
rows, TEST_PROBA = [], {}
for v, runs in VARIANTS.items():
    for s, rid in zip(SEEDS, runs):
        TEST_PROBA[(v, s)] = predict_split(rid, "test")
        tm = evaluate(y_te, TEST_PROBA[(v, s)])
        rows.append({"variant": v, "seed": s, "run_id": rid, "val_macro_f1": load_run(rid)["val_metrics"]["macro_f1"],
                     "val_accuracy": load_run(rid)["val_metrics"]["accuracy"],
                     "test_accuracy": tm["accuracy"], "test_macro_f1": tm["macro_f1"], "test_min_recall": tm["min_recall"]})
tba = pd.DataFrame(rows)
tba.to_csv(RESULTS_DIR / "test_before_after.csv", index=False)
summary = tba.groupby("variant")[["val_macro_f1", "test_accuracy", "test_macro_f1", "test_min_recall"]].agg(["mean", "std"])
display(summary.loc[["before", "after"]].round(3))

# ตัวแทนของแต่ละฝั่ง = seed ที่ Val Macro-F1 สูงสุด (เลือกจาก Val ไม่ใช่ Test)
REP = {v: tba[tba.variant == v].sort_values("val_macro_f1", ascending=False).iloc[0] for v in VARIANTS}
REP_PROBA = {v: TEST_PROBA[(v, int(REP[v].seed))] for v in VARIANTS}
REP_METRICS = {v: evaluate(y_te, REP_PROBA[v]) for v in VARIANTS}

pd.DataFrame([{"variant": v, "class": c, "precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"], "f1": m[f"f1_{c}"]}
              for v, m in REP_METRICS.items() for c in CLASSES]).to_csv(RESULTS_DIR / "per_class_test.csv", index=False)

src_te = split_df["source"][masks["test"]].to_numpy()
per_source = pd.DataFrame([{"variant": v, "source": s, "n": int((src_te == s).sum()),
                            **{k: val for k, val in evaluate(y_te[src_te == s], REP_PROBA[v][src_te == s]).items()
                               if k in ("accuracy", "macro_f1", "min_recall")}}
                           for v in VARIANTS for s in SOURCES if (src_te == s).any()])
per_source.to_csv(RESULTS_DIR / "per_source_test.csv", index=False)
display(per_source.round(3))

pred = {v: p.argmax(1) for v, p in REP_PROBA.items()}
b, c, p_mc = mcnemar(y_te, pred["after"], pred["before"])
sig = pd.DataFrame([{"variant": v, "seed": int(REP[v].seed), "test_macro_f1": REP_METRICS[v]["macro_f1"],
                     **dict(zip(("ci95_low", "ci95_high"), bootstrap_ci(y_te, pred[v], seed=SEED)))} for v in VARIANTS])
sig["mcnemar_after_right_before_wrong"], sig["mcnemar_before_right_after_wrong"], sig["mcnemar_p"] = b, c, p_mc
sig.to_csv(RESULTS_DIR / "significance.csv", index=False)
display(sig.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, v in zip(axes, VARIANTS):
    cm = confusion_matrix(y_te, pred[v], labels=range(len(CLASSES)))
    (RESULTS_DIR / f"confusion_{v}.json").write_text(json.dumps({"classes": CLASSES, "matrix": cm.tolist()}), encoding="utf-8")
    plot_cm(ax, cm)
    ax.set(title=f"{TOP1_NAME} {v} (seed {int(REP[v].seed)}) | Macro-F1 {REP_METRICS[v]['macro_f1']:.3f}",
           xlabel="predicted", ylabel="true")
savefig(fig, "confusion_before_after.png")
plt.show()

after_mean = tba[tba.variant == "after"].test_macro_f1.mean()
am = REP_METRICS["after"]
targets = pd.DataFrame([
    ("Macro-F1 (Test, mean 3 seed)", f"≥ {F1_TARGET}", after_mean, after_mean >= F1_TARGET),
    ("Accuracy (Test, mean 3 seed)", "≥ 85%", tba[tba.variant == "after"].test_accuracy.mean(),
     tba[tba.variant == "after"].test_accuracy.mean() >= 0.85),
    ("Recall ต่ำสุดรายคลาส (Test)", "≥ 0.75", am["min_recall"], am["min_recall"] >= 0.75),
], columns=["ตัวชี้วัด", "เป้า", "ผล", "ผ่าน"]).set_index("ตัวชี้วัด")
display(targets.round(3))

FALLBACK_NEEDED = bool(after_mean < F1_TARGET)
selection["fallback"] = {"needed": FALLBACK_NEEDED, "after_test_macro_f1_mean": float(after_mean), "target": F1_TARGET,
                         "note": "decision looks at Test (the only use of Test before the final report)"}
(RESULTS_DIR / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
if FALLBACK_NEEDED:
    print(f"⚠️ Test Macro-F1 หลังจูน {after_mean:.3f} < {F1_TARGET}: ต้องทำขั้นสำรอง (ConvNeXt V2-Tiny จาก timm)")
else:
    print(f"ผ่านเป้า: Test Macro-F1 หลังจูน {after_mean:.3f} ≥ {F1_TARGET}")

# ผล Test ของทั้ง 16 run ในรอบคัดเลือก: รายงานหลังเลือกโมเดลแล้ว (ไม่ได้ใช้เลือก)
screening_test = pd.DataFrame([{"model": fl.TV_ZOO[a]["name"], "round": r, "candidate": f"{a}|{r}",
                                **{f"test_{k}": v for k, v in evaluate(y_te, predict_split(rid, "test")).items()
                                   if k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "min_recall")}}
                               for (a, r), rid in SCREEN_RUNS.items()])
screening_test.to_csv(RESULTS_DIR / "screening_test.csv", index=False)
display(screening_test.pivot(index="model", columns="round", values="test_macro_f1")[ROUNDS].round(3))
'''

GEN_MD = r'''
## 10b. Generalization: Train vs Val vs Test vs Unseen (ตามสไลด์)

สำหรับทั้ง 16 run ในรอบคัดเลือกและตัวแทนก่อน/หลังจูน วัด Accuracy / Precision / Recall / F1 / Loss บนทุกชุด
(Train วัดใน eval mode ไม่มี augmentation) แล้ววาดกราฟแบบสไลด์ต่อ run:
1. แท่ง Train vs Val vs Test (vs Unseen ถ้ามี) 2. learning curve loss และ accuracy พร้อมข้อความวินิจฉัย overfitting 3. confusion matrix (นับจำนวน) บน Test

ช่องว่างระหว่าง Train กับ Val/Test ที่กว้าง = overfitting ส่วน Unseen ที่ต่ำกว่า Test มาก = โมเดลยังใช้กับรูปต่างแหล่งไม่ได้ดี
'''

GEN = r'''
unseen_df = None
if (UNSEEN_DIR / "sources.csv").exists():
    unseen_df = pd.read_csv(UNSEEN_DIR / "sources.csv")
    unseen_df = unseen_df[unseen_df["class"].isin(CLASSES)].reset_index(drop=True)
    y_un = unseen_df["class"].map(CLASS_TO_IDX).to_numpy()
    print(f"Unseen: {len(unseen_df)} รูป", unseen_df["class"].value_counts().reindex(CLASSES).to_dict())
else:
    print(f"ยังไม่มี {UNSEEN_DIR / 'sources.csv'}: ข้ามส่วน Unseen (สร้างชุด unseen_data แล้วรันส่วน 10b–10c ใหม่)")


def predict_paths(run_id, paths, bs=64):
    ds = fl.FabricDataset(paths, np.zeros(len(paths), dtype=int), train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=NUM_WORKERS)
    return predict_model(load_model(run_id), loader)


def split_metrics(run_id):
    out = {}
    for name, y in (("train", y_tr), ("val", y_va), ("test", y_te)):
        pr = predict_split(run_id, name)
        out[name] = {**evaluate(y, pr), "loss": ce_loss(y, pr)}
    if unseen_df is not None:
        pr = predict_paths(run_id, [UNSEEN_DIR / f for f in unseen_df.file])
        _PROBA_CACHE[(run_id, "unseen")] = pr
        out["unseen"] = {**evaluate(y_un, pr), "loss": ce_loss(y_un, pr)}
    return out


GEN_RUNS = {f"{fl.TV_ZOO[a]['name']} | {r}": rid for (a, r), rid in SCREEN_RUNS.items()}
GEN_RUNS |= {f"{TOP1_NAME} | before (seed {int(REP['before'].seed)})": REP["before"].run_id,
             f"{TOP1_NAME} | after (seed {int(REP['after'].seed)})": REP["after"].run_id}
gen_rows = []
for label, rid in tqdm(GEN_RUNS.items(), desc="generalization"):
    for split, m in split_metrics(rid).items():
        gen_rows.append({"run": label, "run_id": rid, "split": split,
                         **{k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")}})
generalization = pd.DataFrame(gen_rows)
generalization.to_csv(RESULTS_DIR / "generalization.csv", index=False)
display(generalization.pivot(index="run", columns="split", values="macro_f1").round(3))

METRICS = [("accuracy", "Accuracy"), ("macro_precision", "Precision"), ("macro_recall", "Recall"),
           ("macro_f1", "F1-Score"), ("loss", "Loss")]
for label, rid in GEN_RUNS.items():
    g = generalization[generalization.run == label].set_index("split")
    h = pd.DataFrame(load_run(rid)["history"])
    slug = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1, 0.9]})
    for k, split in enumerate(g.index):
        bars = axes[0].bar(np.arange(len(METRICS)) + (k - (len(g) - 1) / 2) * 0.2, [g.loc[split, m] for m, _ in METRICS],
                           width=0.2, label=split)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7)
    axes[0].set(xticks=range(len(METRICS)), xticklabels=[t for _, t in METRICS], title=f"{label}: Generalization")
    axes[0].legend(fontsize=8)
    axes[1].plot(h.epoch, h.train_loss, label="train loss")
    axes[1].plot(h.epoch, h.val_loss, label="val loss")
    axes[1].set(title="Learning curve: loss", xlabel="epoch")
    axes[1].legend(fontsize=8)
    axes[2].plot(h.epoch, h.train_accuracy, label="train acc")
    axes[2].plot(h.epoch, h.val_accuracy, label="val acc")
    best = int(h.loc[h.val_macro_f1.idxmax(), "epoch"])
    for ax in axes[1:3]:
        ax.axvline(best, color="gray", ls="--")
    axes[2].set(title="Learning curve: accuracy", xlabel="epoch")
    axes[2].legend(fontsize=8)
    plot_cm(axes[3], confusion_matrix(y_te, predict_split(rid, "test").argmax(1), labels=range(len(CLASSES))), counts_only=True)
    axes[3].set_title("Confusion matrix (Test)")
    fig.suptitle(diagnose_fit(load_run(rid)["history"], g.loc["train", "accuracy"]), y=1.02, fontsize=10)
    savefig(fig, f"gen_{slug}.png")
    plt.close(fig)
print("บันทึกกราฟต่อ run ไว้ที่", RESULTS_DIR / "gen_*.png")
display(Image.open(RESULTS_DIR / f"gen_{re.sub(r'[^0-9A-Za-z]+', '_', list(GEN_RUNS)[-1]).strip('_')}.png"))
'''

UNSEEN_MD = r'''
## 10c. Unseen data (รูปจากอินเทอร์เน็ต)

รูปผ้าจาก **TextileNet** (Zhong et al., 2023) คลาสละราว 100 รูป เป็นรูปสินค้าจากเว็บ ต่างจากรูปที่ใช้เทรน
คัดกรองโดย AI (ตัดภาพตัดต่อ ภาพที่คนเป็นจุดหลัก และรูปที่ขัดกับ label อย่างชัดเจน) ส่วน label มาจาก TextileNet
ใช้วัดผลอย่างเดียว ไม่ได้ใช้เทรนหรือเลือกโมเดล รัน cell 10b–10c ซ้ำได้หลังเพิ่ม/แก้ `unseen_data/`
'''

UNSEEN = r'''
if unseen_df is None:
    print("ยังไม่มีชุด Unseen: ข้าม")
else:
    rid = REP["after"].run_id
    pr = _PROBA_CACHE.get((rid, "unseen"))
    if pr is None:
        pr = predict_paths(rid, [UNSEEN_DIR / f for f in unseen_df.file])
    pred_u = pr.argmax(1)
    up = unseen_df[["file", "class"]].assign(pred=[CLASSES[i] for i in pred_u], conf=pr.max(1),
                                             **{f"p_{c}": pr[:, i] for i, c in enumerate(CLASSES)})
    up.to_csv(RESULTS_DIR / "unseen_predictions.csv", index=False)
    um = generalization[generalization.split == "unseen"][["run", "accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"]]
    um.to_csv(RESULTS_DIR / "unseen_metrics.csv", index=False)
    m = evaluate(y_un, pr)
    display(pd.DataFrame({c: {"precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"], "f1": m[f"f1_{c}"]}
                          for c in CLASSES}).T.round(3))
    print(f"{TOP1_NAME} หลังจูน บน Unseen: Accuracy {m['accuracy']:.3f}, Macro-F1 {m['macro_f1']:.3f} "
          f"(เทียบ Test {REP_METRICS['after']['macro_f1']:.3f})")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_cm(ax, confusion_matrix(y_un, pred_u, labels=range(len(CLASSES))))
    ax.set_title(f"{TOP1_NAME} after tuning: Unseen (n={len(y_un)})")
    savefig(fig, "unseen_confusion.png")
    plt.show()
'''

ERROR_MD = r'''
## 11. Error analysis

รูปใน Test ที่โมเดลหลังจูนทายผิดโดยมั่นใจสูงสุด 12 รูป และคู่คลาสที่สับสนบ่อย (R3 พบว่าคู่ Cotton–Polyester สับสนกันบ่อยเพราะลายผิวคล้ายกัน)
'''

ERROR = r'''
test_df = split_df[masks["test"]].reset_index(drop=True)
proba = REP_PROBA["after"]
pred_a, conf = proba.argmax(1), proba.max(1)
wrong = np.flatnonzero(pred_a != y_te)
top_wrong = wrong[np.argsort(-conf[wrong])][:12]
print(f"{TOP1_NAME} หลังจูน: ทายผิด {len(wrong)} จาก {len(y_te)} รูปใน Test")
if len(top_wrong):
    ncol = 4
    nrow = int(np.ceil(len(top_wrong) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), top_wrong):
        img = fl.load_pil(DATA_DIR / test_df.path[i])
        img.thumbnail((256, 256))
        ax.imshow(img)
        ax.set_title(f"true {CLASSES[y_te[i]]} -> {CLASSES[pred_a[i]]}\nconf {conf[i]:.0%} | {test_df.source[i]}", fontsize=9)
    fig.suptitle(f"{TOP1_NAME} (after tuning): most confident mistakes (test)")
    savefig(fig, "error_top_confident_mistakes.png")
    plt.show()
cm = confusion_matrix(y_te, pred_a, labels=range(len(CLASSES)))
pairs = pd.DataFrame([{"true": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j]),
                       "% of true class": cm[i, j] / max(cm[i].sum(), 1)}
                      for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j and cm[i, j] > 0])
if len(pairs):
    display(pairs.sort_values("count", ascending=False).head(6).style.format({"% of true class": "{:.1%}"}))
'''

EXPORT_MD = r'''
## 12. Export

บันทึก run "หลังจูน" ที่ Val Macro-F1 สูงสุดเป็น `models/best_model.pth` พร้อม `models/model_meta.json`
แล้วตรวจว่าโหลดผ่าน `fl.load_predictor` (ทางเดียวกับแอป) ได้ผลตรงกับใน notebook
'''

EXPORT = r'''
rid = REP["after"].run_id
shutil.copyfile(RUNS_DIR / f"{rid}.pth", MODELS_DIR / "best_model.pth")
r = load_run(rid)
summ = {v: {"test_macro_f1_mean": float(tba[tba.variant == v].test_macro_f1.mean()),
            "test_macro_f1_std": float(tba[tba.variant == v].test_macro_f1.std(ddof=0)),
            "test_accuracy_mean": float(tba[tba.variant == v].test_accuracy.mean())} for v in VARIANTS}
meta = {"framework": "torch", "model_file": "best_model.pth", "arch": TOP1, "round": TOP1_ROUND,
        "model_name": f"{TOP1_NAME} (รอบ {TOP1_ROUND}, fine-tuned + Optuna)", "classes": CLASSES,
        "img_size": IMG_SIZE, "preprocessing": "RGB, EXIF-rotated, resize shorter side to 256, center crop 224, ImageNet mean/std",
        "params": r["params"], "seed": int(REP["after"].seed), "run_id": rid,
        "val_metrics": r["val_metrics"], "test_metrics": REP_METRICS["after"], "test_before_after": summ,
        "conf_threshold": CONF_THRESHOLD, "infer_ms_per_img": time_inference(rid),
        "fallback_needed": FALLBACK_NEEDED, "n_images": dict(zip(("train", "val", "test"), N_IMAGES)),
        "data_fingerprint": FINGERPRINT, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
(MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# ตรวจว่าโหลดแบบเดียวกับแอปแล้วได้ผลตรงกับ notebook (fp16 + batch ต่างกันทำให้ต่างได้เล็กน้อย)
app_pred = fl.load_predictor(MODELS_DIR, DEVICE)
chk = [DATA_DIR / p for p in split_df["path"][masks["val"]].iloc[:20]]
p_app, p_nb = app_pred.predict_proba(chk), predict_split(rid, "val")[:20]
ok, msg = check_export(p_app, p_nb)
assert ok, f"ผลจาก load_predictor ไม่ตรงกับ notebook: {msg}"
print(f"Export {TOP1_NAME} (seed {meta['seed']}) -> {MODELS_DIR / 'best_model.pth'}; load_predictor ตรงกับ notebook ({msg})")
del app_pred
'''

APP_MD = r'''
## 13. เขียนแอป Streamlit (`app.py`)

แอป 4 หน้า: **เปรียบเทียบ 4 โมเดล × 3 รอบ (16 run)**, **ก่อน vs หลังจูน**, **ทำนายรูป** และ **Unseen data** อ่านเฉพาะไฟล์ใน `results/` และ `models/` ที่ notebook สร้าง
ถ้ายังไม่มีไฟล์ของหน้าใด หน้านั้นจะบอกว่าต้องรัน notebook ส่วนไหน รันเองได้ด้วย `streamlit run app.py`
'''

LAUNCH_MD = r'''
## 14. เปิดแอป

รัน Streamlit แบบ background แล้วเปิด http://localhost:8501 (log อยู่ที่ `results/streamlit.log`) รัน cell นี้ซ้ำจะปิดแอปเดิมแล้วเปิดใหม่
'''

LAUNCH = r'''
import socket, subprocess, webbrowser

APP_URL = f"http://localhost:{APP_PORT}"


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


if not LAUNCH_APP or SMOKE:
    print("ข้ามการเปิดแอป (รันเองได้ด้วย: streamlit run app.py)")
else:
    if "app_proc" in globals() and app_proc.poll() is None:
        app_proc.terminate()
        app_proc.wait(timeout=10)
        time.sleep(1)
    if port_in_use(APP_PORT):
        print(f"พอร์ต {APP_PORT} ถูกใช้อยู่ (อาจเป็นแอปเดิมที่เปิดจากที่อื่น) ปิดโปรแกรมนั้นก่อน หรือเปลี่ยน APP_PORT ใน Config")
    else:
        app_log = open(RESULTS_DIR / "streamlit.log", "w", encoding="utf-8")
        app_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(PROJECT_DIR / "app.py"), "--server.address", "localhost",
             "--server.port", str(APP_PORT), "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=PROJECT_DIR, stdout=app_log, stderr=subprocess.STDOUT)
        for _ in range(120):
            if port_in_use(APP_PORT) or app_proc.poll() is not None:
                break
            time.sleep(0.5)
        if port_in_use(APP_PORT):
            display(HTML(f'<b>FabricSort พร้อมใช้งาน:</b> <a href="{APP_URL}" target="_blank">{APP_URL}</a>'))
            webbrowser.open(APP_URL)
        else:
            print("เปิดแอปไม่สำเร็จ ดู log ที่", RESULTS_DIR / "streamlit.log")
'''


def build(out: Path) -> None:
    lib = (ROOT / "fabric_lib.py").read_text(encoding="utf-8")
    helpers = (ROOT / "tools" / "tuning_helpers.py").read_text(encoding="utf-8")
    app = (ROOT / "app_src" / "app.py").read_text(encoding="utf-8")
    cells = [md(INTRO), md(INSTALL_MD), code(INSTALL), md(CONFIG_MD), code(CONFIG), md(SETUP_MD),
             code("%%writefile fabric_lib.py\n" + lib), md(IMPORTS_MD), code(IMPORTS),
             *reused_cells(), code(SPLIT_CHECK),
             md(TRAIN_MD), code(helpers), code(TRAIN),
             md(SCREEN_MD), code(SCREEN), md(HELPERS_MD), code(SELECT),
             md(BEFORE_MD), code(BEFORE), md(OPTUNA_MD), code(OPTUNA), md(AFTER_MD), code(AFTER),
             md(TEST_MD), code(TEST), md(GEN_MD), code(GEN), md(UNSEEN_MD), code(UNSEEN), md(ERROR_MD), code(ERROR), md(EXPORT_MD), code(EXPORT),
             md(APP_MD), code("%%writefile app.py\n" + app), md(LAUNCH_MD), code(LAUNCH)]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": ".venv (3.14)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out.write_text(nbf.writes(nb), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "fabric_classification.ipynb")
    build(ap.parse_args().out)
