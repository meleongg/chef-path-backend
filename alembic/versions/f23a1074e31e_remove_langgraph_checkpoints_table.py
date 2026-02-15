"""remove_langgraph_checkpoints_table

Revision ID: f23a1074e31e
Revises: a72c1542e137
Create Date: 2025-11-30 14:22:01.879942

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f23a1074e31e"
down_revision: Union[str, Sequence[str], None] = "a72c1542e137"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the old langgraph_checkpoints table (no longer needed - using PostgresSaver managed tables)."""
    op.drop_table("langgraph_checkpoints")


def downgrade() -> None:
    """Recreate langgraph_checkpoints table if needed to rollback."""
    op.create_table(
        "langgraph_checkpoints",
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_id"),
    )
    op.create_index(
        "idx_langgraph_thread_ts",
        "langgraph_checkpoints",
        ["thread_id", sa.text("created_at DESC")],
        unique=False,
    )
