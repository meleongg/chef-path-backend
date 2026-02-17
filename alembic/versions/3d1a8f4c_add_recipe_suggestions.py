"""Add recipe suggestions table and remove weekly plan exclusions

Revision ID: 3d1a8f4c
Revises: 2f8a9c1d
Create Date: 2026-02-16 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3d1a8f4c"
down_revision: Union[str, Sequence[str], None] = "2f8a9c1d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    weekly_plan_columns = {
        column["name"] for column in inspector.get_columns("weekly_plans")
    }
    if "excluded_recipe_ids" in weekly_plan_columns:
        op.drop_column("weekly_plans", "excluded_recipe_ids")

    if "recipe_suggestions" not in inspector.get_table_names():
        op.create_table(
            "recipe_suggestions",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("recipe_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("week_number", sa.Integer(), nullable=False),
            sa.Column("source", sa.String(length=20), nullable=False),
            sa.Column(
                "suggested_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("timezone('utc', now())"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["recipe_id"], ["recipes.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_recipe_suggestions_user_id", "recipe_suggestions", ["user_id"]
        )
        op.create_index(
            "ix_recipe_suggestions_recipe_id", "recipe_suggestions", ["recipe_id"]
        )
        op.create_index(
            "ix_recipe_suggestions_user_suggested_at",
            "recipe_suggestions",
            ["user_id", "suggested_at"],
        )
        op.create_index(
            "ix_recipe_suggestions_user_recipe",
            "recipe_suggestions",
            ["user_id", "recipe_id"],
        )


def downgrade() -> None:
    op.drop_index("ix_recipe_suggestions_user_recipe", table_name="recipe_suggestions")
    op.drop_index(
        "ix_recipe_suggestions_user_suggested_at", table_name="recipe_suggestions"
    )
    op.drop_index("ix_recipe_suggestions_recipe_id", table_name="recipe_suggestions")
    op.drop_index("ix_recipe_suggestions_user_id", table_name="recipe_suggestions")
    op.drop_table("recipe_suggestions")

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
