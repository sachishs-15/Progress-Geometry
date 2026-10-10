"""Fill report_template.md's {{T:name}} placeholders from make_tables.py -> REPORT.md; also writes results/seed_sd_cls_family.json."""
import json
import re
from pathlib import Path

import numpy as np

import make_tables as mt
from metrics_batch import PER_CLIP_KEYS

HERE = Path(__file__).resolve().parent


def seed_sd():
    seeds = [np.load(HERE / f"results/noise_study/vicreg_s{s}.perclip.npz") for s in range(1, 6)]
    sd = {m: float(np.std([s[m][s["keep"]].mean() for s in seeds], ddof=1)) for m in PER_CLIP_KEYS}
    (HERE / "results/seed_sd_cls_family.json").write_text(json.dumps(sd, indent=1))


if __name__ == "__main__":
    seed_sd()
    text = (HERE / "report_template.md").read_text()
    text = re.sub(r"\{\{T:(\w+)\}\}", lambda m: mt.TABLES[m.group(1)](), text)
    (HERE / "REPORT.md").write_text(text)
    print("REPORT.md:", len(text.splitlines()), "lines")
