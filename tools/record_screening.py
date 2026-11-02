"""บันทึกผลคัดกรองรูป Unseen (ทีละแผ่น) ลง decisions.csv

stdin: JSON {"<stem>": ["keep"|"drop", "คำอธิบาย"], ...}
รัน: ... | python -I tools/record_screening.py <decisions.csv>
"""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = json.loads(sys.stdin.buffer.read().decode("utf-8"))
old = {}
if out.exists():
    with open(out, encoding="utf-8") as f:
        old = {r["stem"]: r for r in csv.DictReader(f)}
for stem, (decision, desc) in rows.items():
    assert decision in ("keep", "drop"), stem
    old[stem] = {"stem": stem, "decision": decision, "description": desc}
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["stem", "decision", "description"])
    w.writeheader()
    w.writerows(sorted(old.values(), key=lambda r: r["stem"]))
k = sum(r["decision"] == "keep" for r in old.values())
sys.stdout.reconfigure(encoding="utf-8")
print(f"บันทึก {len(rows)} รูป | รวม {len(old)} รูป, เก็บ {k}")
