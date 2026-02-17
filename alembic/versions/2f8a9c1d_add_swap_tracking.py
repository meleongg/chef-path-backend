"""Add swap tracking fields to weekly_plans

Revision ID: 2f8a9c1d
Revises: ae7f8b2c
Create Date: 2026-02-16 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "2f8a9c1d"
down_revision: Union[str, Sequence[str], None] = "ae7f8b2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "weekly_plans",
        sa.Column(
            "swap_count", sa.Integer(), default=0, nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "weekly_plans",
        sa.Column(
            "excluded_recipe_ids",
            sa.Text(),
            default="[]",
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("weekly_plans", "excluded_recipe_ids")
    op.drop_column("weekly_plans", "swap_count")
