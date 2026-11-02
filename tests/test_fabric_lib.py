import io
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

import fabric_lib as fl
from conftest import ROOT

ARCHS = ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"]


def _jpeg_with_orientation(w=60, h=40, orientation=6):
    img = Image.new("RGB", (w, h), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_load_pil_handles_modes_and_exif():
    for src in (Image.new("RGBA", (30, 20), (0, 0, 0, 0)), Image.new("L", (30, 20), 128)):
        assert fl.load_pil(src).mode == "RGB"
    out = fl.load_pil(_jpeg_with_orientation(60, 40))
    assert out.mode == "RGB" and out.size == (40, 60)


@pytest.mark.parametrize("arch", ARCHS)
def test_build_model_shapes(arch):
    model = fl.build_model(arch, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 4)
    head = list(fl.head_params(model))
    assert head and all(p.shape[0] in (4,) for p in head if p.ndim == 2)


def test_set_trainable():
    model = fl.build_model("resnet50", pretrained=False)
    fl.set_trainable(model, 0.0)
    head_ids = {id(p) for p in fl.head_params(model)}
    assert all(not p.requires_grad for p in model.parameters() if id(p) not in head_ids)
    assert all(p.requires_grad for p in fl.head_params(model))
    fl.set_trainable(model, 0.3)
    leaves = fl.backbone_units(model)
    n_top = int(np.ceil(0.3 * len(leaves)))
    for i, m in enumerate(leaves):
        expect = i >= len(leaves) - n_top and not isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        assert all(p.requires_grad == expect for p in m.parameters(recurse=False))
    fl.train_mode(model)
    assert model.training
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def _write_images(tmp_path, n=5):
    paths = []
    for i in range(n):
        p = tmp_path / f"im{i}.png"
        Image.fromarray(np.random.randint(0, 255, (300, 260, 3), dtype="uint8")).save(p)
        paths.append(p)
    return paths


def test_dataset_item(tmp_path):
    paths = _write_images(tmp_path)
    for train, aug in ((False, "none"), (True, "none"), (True, "v1"), (True, "v2"), (True, "texture")):
        ds = fl.FabricDataset(paths, [0, 1, 2, 3, 0], train=train, aug=aug)
        x, y = ds[1]
        assert len(ds) == 5 and x.shape == (3, 224, 224) and x.dtype == torch.float32 and y == 1


def test_mix_batch():
    rng = np.random.default_rng(0)
    x = torch.rand(8, 3, 224, 224)
    y = torch.eye(4)[torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])]
    soft = False
    for _ in range(10):
        xm, ym = fl.mix_batch(x, y, rng)
        assert xm.shape == x.shape and torch.allclose(ym.sum(1), torch.ones(8))
        soft |= bool(((ym > 0) & (ym < 1)).any())
    assert soft


def _save_model_dir(tmp_path):
    model = fl.build_model("efficientnet_b0", pretrained=False).eval()
    torch.save(model.state_dict(), tmp_path / "best_model.pth")
    (tmp_path / "model_meta.json").write_text(json.dumps(
        {"framework": "torch", "arch": "efficientnet_b0", "model_file": "best_model.pth",
         "classes": fl.CLASSES, "img_size": 224}))
    return model


def test_torch_roundtrip_matches(tmp_path):
    model = _save_model_dir(tmp_path)
    imgs = _write_images(tmp_path, 3)
    ref = fl.TorchPredictor(model, "cpu").predict_proba(imgs)
    assert np.allclose(ref.sum(1), 1.0, atol=1e-4)
    code = ("import sys,json;sys.path.insert(0,sys.argv[1]);import fabric_lib as fl;"
            "p=fl.load_predictor(sys.argv[2],'cpu');print(json.dumps(p.predict_proba(sys.argv[3:]).tolist()))")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT), str(tmp_path), *map(str, imgs)],
                         capture_output=True, text=True, check=True)
    got = np.array(json.loads(out.stdout.strip().splitlines()[-1]))
    assert np.abs(got - ref).max() < 1e-4


def test_load_predictor_rejects_non_torch(tmp_path):
    (tmp_path / "model_meta.json").write_text(json.dumps({"framework": "keras", "model_file": "x.keras"}))
    with pytest.raises(ValueError):
        fl.load_predictor(tmp_path, "cpu")


@pytest.mark.parametrize("arch", ARCHS)
def test_set_trainable_freezes_all_backbone_params(arch):
    """review #1: ConvNeXt layer_scale อยู่บน CNBlock (ไม่ใช่ leaf) ต้องถูกแช่แข็งด้วย"""
    model = fl.build_model(arch, pretrained=False)
    head_ids = {id(p) for p in fl.head_params(model)}
    fl.set_trainable(model, 0.0)
    assert [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in head_ids] == []
    fl.set_trainable(model, 0.3)
    units = fl.backbone_units(model)
    n_top = int(np.ceil(0.3 * len(units)))
    allowed = {id(p) for m in units[len(units) - n_top:] for p in m.parameters(recurse=False)} | head_ids
    assert all(id(p) in allowed for p in model.parameters() if p.requires_grad)


def test_load_pil_16bit_gray_not_white():
    """minor #8: PNG 16-bit ขาวดำต้องย่อสเกลเป็น 8-bit ไม่ใช่กลายเป็นสีขาวทั้งรูป"""
    img = Image.fromarray(np.full((20, 30), 30000, dtype=np.uint16))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = np.asarray(fl.load_pil(buf.getvalue()))
    assert out.shape == (20, 30, 3)
    assert 100 < out.mean() < 135  # 30000 / 257 ≈ 117


def test_load_pil_large_jpeg_downscaled_on_decode():
    """minor #8: รูป JPEG ขนาดใหญ่จากมือถือถูกย่อตอนถอดรหัส (ประหยัดหน่วยความจำ) แต่ยังใหญ่พอสำหรับ 224"""
    buf = io.BytesIO()
    Image.new("RGB", (6000, 4000), (120, 60, 30)).save(buf, "JPEG")
    out = fl.load_pil(buf.getvalue())
    assert out.mode == "RGB" and 256 <= min(out.size) <= 2048 + 1


def test_load_pil_too_large_raises_clear_error(monkeypatch):
    """minor #8: รูปใหญ่เกินขีดจำกัดต้องได้ข้อความที่ชัดเจน (ไม่ใช่ 'ไฟล์เสีย')"""
    monkeypatch.setattr(fl, "MAX_PIXELS", 1000)
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "PNG")
    with pytest.raises(fl.ImageTooLargeError):
        fl.load_pil(buf.getvalue())


def test_aug_none_equals_eval(tmp_path):
    """R3: รอบ none = ไม่ทำ augmentation (เหมือนตอนวัดผลทุกประการ)"""
    paths = _write_images(tmp_path, 2)
    a = fl.FabricDataset(paths, [0, 1], train=True, aug="none")[0][0]
    b = fl.FabricDataset(paths, [0, 1], train=False)[0][0]
    assert torch.equal(a, b)


def test_aug_v1_v2_random(tmp_path):
    paths = _write_images(tmp_path, 1)
    for aug in ("v1", "v2", "texture"):
        ds = fl.FabricDataset(paths, [0], train=True, aug=aug)
        assert not torch.equal(ds[0][0], ds[0][0])


def test_unknown_aug_raises():
    with pytest.raises(ValueError):
        fl.build_transforms(224, train=True, aug="base")


@pytest.mark.parametrize("arch", ARCHS)
def test_imagenet_metrics(arch):
    """R7 SMOKE: ดึง ImageNet Top-1/Top-5 ของ weights ที่ใช้ (ไม่ต้องดาวน์โหลด)"""
    top1, top5 = fl.imagenet_metrics(arch)
    assert 70 < top1 < 90 and top1 < top5 < 100


def test_texture_aug_has_no_blur_noise_or_tilt():
    """รอบ texture: ครอปสุ่ม + หมุน 90° + กลับด้าน + ปรับสีเล็กน้อย ไม่มี blur / noise / หมุนเอียง"""
    from torchvision import transforms as T

    names = [type(t).__name__ for t in fl.build_transforms(224, train=True, aug="texture").transforms]
    assert "RandomResizedCrop" in names and "RandomRot90" in names and "RandomHorizontalFlip" in names
    assert "ColorJitter" in names
    assert not {"RandomAffine", "AddGaussianNoise", "RandomApply", "GaussianBlur"} & set(names)


def test_dataset_retries_transient_oserror(tmp_path, monkeypatch):
    """OneDrive อาจอ่านไฟล์ไม่ได้ชั่วคราว (OSError Errno 22): ลองใหม่ก่อนยอมแพ้"""
    paths = _write_images(tmp_path, 1)
    ds = fl.FabricDataset(paths, [0], train=False)
    real, calls = fl.load_pil, {"n": 0}

    def flaky(src):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(22, "Invalid argument")
        return real(src)

    monkeypatch.setattr(fl, "load_pil", flaky)
    monkeypatch.setattr(fl, "RETRY_SLEEP_S", 0)
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and calls["n"] == 3
