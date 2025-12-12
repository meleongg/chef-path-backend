"""Add AI-augmented metadata fields to Recipe model

Revision ID: b9e6bed410c5
Revises: f23a1074e31e
Create Date: 2025-12-11 23:53:24.412466

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b9e6bed410c5'
down_revision: Union[str, Sequence[str], None] = 'f23a1074e31e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new AI-augmented metadata fields to recipes table
    op.add_column('recipes', sa.Column('dietary_tags', sa.Text(), nullable=True))
    op.add_column('recipes', sa.Column('allergens', sa.Text(), nullable=True))
    op.add_column('recipes', sa.Column('portion_size', sa.String(length=50), nullable=True))
    op.add_column('recipes', sa.Column('prep_time_minutes', sa.Integer(), nullable=True))
    op.add_column('recipes', sa.Column('cook_time_minutes', sa.Integer(), nullable=True))
    op.add_column('recipes', sa.Column('skill_level_validated', sa.String(length=20), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove AI-augmented metadata fields from recipes table
    op.drop_column('recipes', 'skill_level_validated')
    op.drop_column('recipes', 'cook_time_minutes')
    op.drop_column('recipes', 'prep_time_minutes')
    op.drop_column('recipes', 'portion_size')
    op.drop_column('recipes', 'allergens')
    op.drop_column('recipes', 'dietary_tags')
