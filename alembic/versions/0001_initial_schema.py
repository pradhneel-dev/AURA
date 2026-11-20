"""Create the initial tenant-scoped AURA schema."""
from alembic import op
from backend.database.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # The metadata is versioned by this initial baseline. Subsequent changes should
    # use explicit Alembic operations in new revisions.
    Base.metadata.create_all(bind=op.get_bind())


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())
