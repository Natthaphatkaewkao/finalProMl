"""รัน notebook พร้อมเฝ้าการ์ดจอ: หยุดทันทีเมื่อร้อน/ทำงานนาน/ใช้แบต/การ์ดจอหลุด แล้วพักจนเย็นก่อนรันต่อ

รัน (แยกจากเซสชัน): .venv\\Scripts\\python.exe -I tools\\gpu_watchdog.py
log: results/watchdog.log (บรรทัดที่ขึ้นต้นด้วย STOP / RESUME / DONE / FATAL / START)
notebook รันต่อจากจุดเดิมได้เพราะทุก run ถูก cache และ Optuna เก็บใน results/optuna.db
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "watchdog.log"

TEMP_CRITICAL = 90        # °C หยุดทันที
TEMP_HIGH = 85            # °C หยุดถ้าร้อนต่อเนื่องนาน HOT_SECONDS
HOT_SECONDS = 60
MAX_RUN_SECONDS = 90 * 60  # เทรนต่อเนื่องได้สูงสุด 90 นาทีแล้วพัก
COOL_TEMP = 50            # พักจนเย็นถึง °C นี้
MIN_REST_SECONDS = 15 * 60
MAX_RESTARTS = 4
POLL_SECONDS = 10
NOTEBOOK = "fabric_classification.ipynb"


def decide(temp_c, hot_s: float, run_s: float, on_ac: bool, gpu_ok: bool) -> str | None:
    """เหตุผลที่ต้องหยุด หรือ None ถ้ารันต่อได้ (เรียงตามความรุนแรง)"""
    if not gpu_ok:
        return "gpu_lost"
    if temp_c is not None and temp_c >= TEMP_CRITICAL:
        return "temp_critical"
    if hot_s >= HOT_SECONDS:
        return "temp_high"
    if MAX_RUN_SECONDS is not None and run_s >= MAX_RUN_SECONDS:
        return "run_too_long"
    if not on_ac:
        return "on_battery"
    return None


def apply_args(argv) -> None:
    """--no-time-limit: พักตามอุณหภูมิเท่านั้น, --min-rest-min N: พักอย่างน้อย N นาทีหลังหยุด"""
    import argparse

    global MAX_RUN_SECONDS, MIN_REST_SECONDS, NOTEBOOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-time-limit", action="store_true")
    ap.add_argument("--min-rest-min", type=float, default=MIN_REST_SECONDS / 60)
    ap.add_argument("--notebook", default=NOTEBOOK)
    a = ap.parse_args(argv)
    if a.no_time_limit:
        MAX_RUN_SECONDS = None
    MIN_REST_SECONDS = a.min_rest_min * 60
    NOTEBOOK = a.notebook


def cooldown_ready(temp_c, rest_s: float) -> bool:
    return temp_c is not None and temp_c <= COOL_TEMP and rest_s >= MIN_REST_SECONDS


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read_gpu():
    """(อุณหภูมิ °C, การ์ดจอใช้ได้ไหม)"""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return int(t), True
    except Exception:
        return None, False


def on_ac_power() -> bool:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out in ("", "2") or "2" in out.split()  # 2 = เสียบสายชาร์จ (ไม่มีแบต = ถือว่าใช้ไฟบ้าน)
    except Exception:
        return True


def start_notebook():
    cmd = [str(ROOT / ".venv" / "Scripts" / "jupyter.exe"), "nbconvert", "--to", "notebook", "--execute", "--inplace",
           NOTEBOOK, "--ExecutePreprocessor.timeout=-1"]
    log_name = "run_full.log" if NOTEBOOK == "fabric_classification.ipynb" else f"run_{Path(NOTEBOOK).stem}.log"
    out = open(ROOT / "results" / log_name, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def kill_tree(proc) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def main() -> int:
    restarts = 0
    while True:
        log(f"START {NOTEBOOK} (ครั้งที่ {restarts + 1})")
        proc, t0, hot_since, last_ac_check, ac = start_notebook(), time.time(), None, 0.0, True
        reason = None
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            temp, ok = read_gpu()
            now = time.time()
            if temp is not None and temp >= TEMP_HIGH:
                hot_since = hot_since or now
            else:
                hot_since = None
            if now - last_ac_check > 30:
                ac, last_ac_check = on_ac_power(), now
            reason = decide(temp, (now - hot_since) if hot_since else 0, now - t0, ac, ok)
            if reason:
                kill_tree(proc)
                log(f"STOP reason={reason} temp={temp} run_min={(now - t0) / 60:.0f}")
                break
        if reason is None:
            code = proc.returncode
            log(f"DONE exit={code}" if code == 0 else f"FATAL notebook exit={code} (ดู results/run_full.log)")
            return code
        if reason == "gpu_lost":
            log("FATAL gpu_lost: ต้องรีสตาร์ทเครื่อง (Start > Power > Restart)")
            return 2
        restarts += 1
        if restarts > MAX_RESTARTS:
            log("FATAL หยุดเกินจำนวนครั้งที่กำหนด")
            return 3
        rest0 = time.time()
        while True:
            time.sleep(30)
            temp, ok = read_gpu()
            if not ok:
                log("FATAL gpu_lost ระหว่างพัก")
                return 2
            if cooldown_ready(temp, time.time() - rest0) and on_ac_power():
                log(f"RESUME temp={temp} rest_min={(time.time() - rest0) / 60:.0f}")
                break


if __name__ == "__main__":
    apply_args(sys.argv[1:])
    sys.exit(main())
