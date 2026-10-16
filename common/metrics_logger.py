# -*- coding: utf-8 -*-
"""
metrics_logger.py
简单 CSV 记录器：首次写入表头，之后追加。
"""

import csv
import os
from typing import List


class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        self.f = open(filepath, "a", encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)

        if not file_exists:
            self.w.writeheader()
            self.f.flush()

    def log(self, **kw):
        row = {k: kw.get(k, "") for k in self.fieldnames}
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
