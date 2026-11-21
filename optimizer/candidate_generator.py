from __future__ import annotations

from typing import Any


def candidates_from_profiles(profiles: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    """Only profiles with direct measurements are candidates; aggregation is kept explicit."""
    return [
        profile for profile in profiles
        if (
            task_type == "all"
            or profile.get("task_type") in (task_type, "all", None)
            or (task_type == "text_to_sql" and profile.get("task_type") == "spider_text_to_sql")
        )
        and profile.get("sample_count", 0) > 0
    ]
