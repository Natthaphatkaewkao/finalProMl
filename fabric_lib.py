"""โค้ดที่ใช้ร่วมกันระหว่าง fabric_classification.ipynb และ app.py

ไฟล์นี้ถูกสร้างจาก notebook (ส่วน 1.1) ให้แก้ใน notebook แล้วรัน cell นั้นใหม่ ไม่ต้องแก้ไฟล์นี้โดยตรง
"""
from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = ["Cotton", "Polyester", "Silk", "Viscose"]
IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------- โหลดรูป ----------
MAX_PIXELS = 250_000_000      # รูปใหญ่กว่านี้ (เช่นเกิน 250 MP) ไม่รับ เพื่อกันหน่วยความจำเต็ม
DECODE_MIN_SIDE = 1024        # JPEG ใหญ่ ๆ ย่อตอนถอดรหัสโดยด้านสั้นยังไม่ต่ำกว่านี้ (ใหญ่พอสำหรับ 224) ประหยัดหน่วยความจำ


class ImageTooLargeError(ValueError):
    """รูปมีจำนวนพิกเซลเกิน MAX_PIXELS"""


def load_pil(src) -> Image.Image:
    """path / bytes / PIL.Image -> PIL RGB ที่หมุนตาม EXIF แล้ว (พื้นโปร่งใสเติมเป็นสีขาว, 16-bit ย่อเป็น 8-bit)"""
    if isinstance(src, Image.Image):
        img = src
    else:
        Image.MAX_IMAGE_PIXELS = None  # ตรวจขนาดเองด้านล่าง ให้ข้อความ error ที่ชัดเจนกว่า
        img = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        w, h = img.size
        if w * h > MAX_PIXELS:
            raise ImageTooLargeError(f"{w}×{h} พิกเซล เกินขีดจำกัด {MAX_PIXELS / 1e6:.0f} MP")
        if img.format == "JPEG" and min(w, h) >= 2 * DECODE_MIN_SIDE:
            img.draft("RGB", (DECODE_MIN_SIDE, DECODE_MIN_SIDE))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):  # ภาพขาวดำความลึกสูง -> 8-bit
        arr = np.asarray(img, dtype=np.float32)
        top = 65535.0 if img.mode.startswith("I;16") or arr.max() > 255 else 255.0
        img = Image.fromarray(np.clip(arr * (255.0 / top), 0, 255).astype(np.uint8), "L")
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return img.convert("RGB")


# ---------- Augmentation (3 รอบตามสไลด์: none / v1 / v2) ----------
AUG_MODES = ("none", "v1", "v2", "texture")
RETRY_SLEEP_S = 1.0   # รออ่านไฟล์ใหม่เมื่อ OneDrive อ่านไม่ได้ชั่วคราว


class RandomRot90:
    """หมุน 0/90/180/270 องศาแบบสุ่ม (ลายผ้าไม่มีด้านบนที่แน่นอน และไม่เกิดขอบดำเหมือนการหมุนมุมอื่น)"""

    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k, expand=True) if k else img


class AddGaussianNoise:
    """บวก noise แบบ Gaussian (sigma เทียบกับช่วง 0–1) กับ tensor ก่อน normalize แล้วตัดให้อยู่ใน 0–1"""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x):
        import torch

        return (x + torch.randn_like(x) * self.sigma).clamp(0.0, 1.0)


def build_transforms(img_size: int = IMG_SIZE, train: bool = False, aug: str = "none"):
    """eval / none: ย่อด้านสั้นเป็น 256 แล้ว center crop 224
    v1: กลับด้านแนวนอน, หมุน ±10.8° (3%), เลื่อน 5%, noise 2%, blur p=0.30
    v2: กลับด้านแนวนอน, หมุน ±21.6° (6%), เลื่อน 8%, ความสว่าง/คอนทราสต์ ±10%, noise 5%, blur p=0.50
    texture: แบบรักษาพื้นผิวผ้า สุ่มครอป 40–100% + หมุนทีละ 90° + กลับด้าน + ปรับสีเล็กน้อย (ไม่มี blur/noise/หมุนเอียง)"""
    from torchvision import transforms as T

    if aug not in AUG_MODES:
        raise ValueError(f"aug ต้องเป็น {AUG_MODES} (ได้ {aug!r})")
    resize = T.Resize(int(round(img_size * 256 / 224)), antialias=True)
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train or aug == "none":
        return T.Compose([resize, T.CenterCrop(img_size), T.ToTensor(), norm])
    if aug == "texture":
        return T.Compose([T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.8, 1.25), antialias=True),
                          RandomRot90(), T.RandomHorizontalFlip(),
                          T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02), T.ToTensor(), norm])
    v2 = aug == "v2"
    ops = [resize, T.RandomCrop(img_size), T.RandomHorizontalFlip(),
           T.RandomAffine(degrees=21.6 if v2 else 10.8, translate=(0.08, 0.08) if v2 else (0.05, 0.05))]
    if v2:
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.1))
    ops += [T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5 if v2 else 0.3),
            T.ToTensor(), AddGaussianNoise(0.05 if v2 else 0.02), norm]
    return T.Compose(ops)


class FabricDataset:
    """Dataset แบบ map-style ที่โหลดรูปจากดิสก์ทีละรูป คืน (tensor 3×224×224, label)"""

    def __init__(self, paths, labels, train: bool = False, aug: str = "none", img_size: int = IMG_SIZE):
        self.paths = [str(p) for p in paths]
        self.labels = [int(y) for y in labels]
        self.transform = build_transforms(img_size, train, aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import time

        for attempt in range(3):  # OneDrive อาจอ่านไฟล์ไม่ได้ชั่วคราว (OSError) ลองใหม่สูงสุด 3 ครั้ง
            try:
                img = globals()["load_pil"](self.paths[i])
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(RETRY_SLEEP_S)
        return self.transform(img), self.labels[i]


def mix_batch(x, y, rng: np.random.Generator):
    """MixUp (alpha 0.2) หรือ CutMix (alpha 1.0) อย่างใดอย่างหนึ่งต่อ batch; y เป็น one-hot/soft"""
    import torch

    perm = torch.from_numpy(rng.permutation(len(x))).to(x.device)
    if rng.random() < 0.5:
        lam = float(rng.beta(0.2, 0.2))
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    lam = float(rng.beta(1.0, 1.0))
    h, w = x.shape[-2:]
    ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
    t, b = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    l, r = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[..., t:b, l:r] = x[perm][..., t:b, l:r]
    lam = 1 - (b - t) * (r - l) / (h * w)
    return x, lam * y + (1 - lam) * y[perm]


# ---------- โมเดล: 4 CNN จาก torchvision ----------
TV_ZOO = {
    "resnet50": {"name": "ResNet50", "weights": "IMAGENET1K_V2"},
    "densenet121": {"name": "DenseNet121", "weights": "IMAGENET1K_V1"},
    "efficientnet_b0": {"name": "EfficientNet-B0", "weights": "IMAGENET1K_V1"},
    "convnext_tiny": {"name": "ConvNeXt-Tiny", "weights": "IMAGENET1K_V1"},
}


def build_model(arch: str, dropout: float = 0.3, pretrained: bool = True, num_classes: int = len(CLASSES)):
    """โมเดล torchvision ที่เปลี่ยนชั้นสุดท้ายเป็น Dropout + Linear(num_classes)

    pretrained=True ดาวน์โหลด weights ของ ImageNet (ตอนเทรน), False = โครงเปล่า (ตอนโหลดในแอป)"""
    import torch.nn as nn
    from torchvision import models

    if arch not in TV_ZOO:
        raise ValueError(f"ไม่รู้จัก arch {arch!r} (รองรับ {list(TV_ZOO)})")
    weights = TV_ZOO[arch]["weights"] if pretrained else None
    model = getattr(models, arch)(weights=weights)
    if arch == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "densenet121":
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes))
    elif arch == "efficientnet_b0":
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[-1].in_features, num_classes))
    else:  # convnext_tiny: classifier = LayerNorm2d, Flatten, Linear -> คง LayerNorm ไว้
        ln, flat, lin = model.classifier
        model.classifier = nn.Sequential(ln, flat, nn.Dropout(dropout), nn.Linear(lin.in_features, num_classes))
    model.arch = arch
    return model


def imagenet_metrics(arch: str) -> tuple[float, float]:
    """ความแม่นยำ ImageNet Top-1 / Top-5 (%) ของ weights ที่ใช้ อ่านจาก metadata ของ torchvision (ไม่ต้องดาวน์โหลด)"""
    from torchvision.models import get_model_weights

    m = get_model_weights(arch)[TV_ZOO[arch]["weights"]].meta["_metrics"]["ImageNet-1K"]
    return float(m["acc@1"]), float(m["acc@5"])


def _head(model):
    return model.fc if model.arch == "resnet50" else model.classifier


def head_params(model):
    """พารามิเตอร์ของหัวจำแนก (สำหรับ convnext รวม LayerNorm ก่อนหัวด้วย)"""
    return list(_head(model).parameters())


def backbone_units(model):
    """โมดูลของ backbone ที่มีพารามิเตอร์ของตัวเอง (recurse=False) เรียงตามลำดับที่ข้อมูลไหลผ่าน
    รวมโมดูลที่ไม่ใช่ใบด้วย เช่น CNBlock ของ ConvNeXt ที่เก็บ layer_scale ไว้ที่ตัวบล็อก"""
    head_ids = {id(m) for m in _head(model).modules()}
    return [m for m in model.modules() if id(m) not in head_ids and list(m.parameters(recurse=False))]


def set_trainable(model, unfreeze_frac: float) -> None:
    """0 = แช่แข็ง backbone ทั้งหมด, อื่น ๆ = เปิดโมดูลบนสุด ceil(frac × จำนวน) ส่วน BatchNorm แช่แข็งเสมอ
    หัวจำแนกเทรนได้เสมอ"""
    from torch.nn.modules.batchnorm import _BatchNorm

    for p in model.parameters():
        p.requires_grad = False
    units = backbone_units(model)
    n_top = 0 if unfreeze_frac <= 0 else math.ceil(unfreeze_frac * len(units))
    for m in units[len(units) - n_top:]:
        if not isinstance(m, _BatchNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    for p in head_params(model):
        p.requires_grad = True


def train_mode(model):
    """model.train() แต่ให้ BatchNorm อยู่ใน eval mode (ใช้ค่าสถิติจาก ImageNet ไม่อัปเดต)"""
    from torch.nn.modules.batchnorm import _BatchNorm

    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
    return model


# ---------- Predictor: ใช้ทั้งตอนวัดผลใน notebook และในแอป ----------
class TorchPredictor:
    kind = "torch"

    def __init__(self, model, device, classes=CLASSES, img_size: int = IMG_SIZE, batch_size: int = 32):
        import torch

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tf = build_transforms(img_size, train=False)
        self.classes = list(classes)
        self.batch_size = batch_size

    def predict_proba(self, images) -> np.ndarray:
        import torch

        images = list(images)
        use_amp = self.device.type == "cuda"
        out = []
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            for i in range(0, len(images), self.batch_size):
                x = torch.stack([self.tf(load_pil(im)) for im in images[i:i + self.batch_size]]).to(self.device)
                out.append(torch.softmax(self.model(x).float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype="float32")


def load_predictor(model_dir, device=None) -> TorchPredictor:
    """โหลดโมเดลที่ดีที่สุดตาม model_meta.json"""
    import torch

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    if meta.get("framework") != "torch":
        raise ValueError("model_meta.json นี้ไม่ใช่โมเดลของเวอร์ชันนี้ ให้รัน notebook ส่วนที่ 12 (Export) ใหม่")
    model = build_model(meta["arch"], pretrained=False, num_classes=len(meta.get("classes", CLASSES)))
    model.load_state_dict(torch.load(model_dir / meta["model_file"], map_location="cpu", weights_only=True))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = TorchPredictor(model, device, meta.get("classes", CLASSES), meta.get("img_size", IMG_SIZE))
    pred.meta = meta
    return pred
