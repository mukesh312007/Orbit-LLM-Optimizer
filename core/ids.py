from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def new_experiment_id(prefix: str = "exp") -> str:
    """Stable, human-readable identifier without relying on an unsafe timestamp alone."""
    return f"{prefix}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
