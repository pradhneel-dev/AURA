"""Add tenant-scoped LLM usage/cost records."""
from alembic import op
from backend.database.models import LLMUsage

revision = "0002_llm_usage"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    LLMUsage.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade():
    LLMUsage.__table__.drop(bind=op.get_bind(), checkfirst=True)
