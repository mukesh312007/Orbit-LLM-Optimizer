"""Initialize MySQL schema after DATABASE_URL is set; never falls back to SQLite."""

from config.settings import settings
from database.repository import ExperimentRepository


if __name__ == "__main__":
    if not settings.database_url:
        raise SystemExit("Set DATABASE_URL (see .env.example).")
    repository = ExperimentRepository(settings.database_url)
    repository.initialize_schema()
    print("MySQL schema initialized.")
