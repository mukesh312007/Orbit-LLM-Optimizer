"""Initial schema migration, usable from a Python migration runner."""
from database.schema import Base


def upgrade(engine) -> None:
    Base.metadata.create_all(engine)


def downgrade(engine) -> None:
    Base.metadata.drop_all(engine)
