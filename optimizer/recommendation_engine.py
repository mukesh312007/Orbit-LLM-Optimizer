from __future__ import annotations

from typing import Any


def explain(selected: dict[str, Any] | None, preference: str, no_feasible: bool = False) -> str:
    if not selected:
        return "No tested configuration meets all supplied constraints. Alternatives identify their violated constraints."
    source = "measured" if selected.get("measurement_kind", "measured") == "measured" else "predicted"
    return f"Selected {selected.get('configuration_id')} using {source} profile data ({selected.get('sample_count', 0)} samples) for {preference}-first optimization. Historical performance is not a guarantee."
