from __future__ import annotations

from core.settings import settings
from database.schema import create_schema

if __name__ == "__main__":
    if not settings.database_url or not settings.database_url.startswith("mysql+"):
        raise SystemExit("Set DATABASE_URL to a MySQL URL in .env; SQLite is reserved for tests.")
    create_schema(settings.database_url)
    print("MySQL schema initialized.")
