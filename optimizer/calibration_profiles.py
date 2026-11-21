"""Read the latest offline quality calibration for live weighted selection."""
from __future__ import annotations

import json
from typing import Any

from core.settings import ROOT

CALIBRATION_PATH = ROOT / "logs" / "spider_calibration.jsonl"


def latest_spider_quality(model_id: str, configuration_id: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    try:
        with CALIBRATION_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("model_id") == model_id and record.get("configuration", {}).get("configuration_id") == configuration_id:
                    latest = record
    except FileNotFoundError:
        return None
    return latest
