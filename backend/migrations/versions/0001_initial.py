"""Initial immutable snapshots, media registry and durable queue."""

from alembic import op
from app.models.db import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.create_all(op.get_bind())


def downgrade():
    raise RuntimeError("Destructive downgrade is intentionally disabled; restore a database backup.")
