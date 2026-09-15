from alembic import context
from app.config import Settings
from app.models.db import Base, database

settings = Settings()
settings.prepare()
engine, _ = database(settings.data_dir)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
