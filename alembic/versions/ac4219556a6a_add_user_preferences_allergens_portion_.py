"""Add user preferences: allergens, portion size, time constraints

Revision ID: ac4219556a6a
Revises: b9e6bed410c5
Create Date: 2025-12-12 00:02:43.985193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ac4219556a6a'
down_revision: Union[str, Sequence[str], None] = 'b9e6bed410c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add user preference fields
    op.add_column('users', sa.Column('dietary_restrictions', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('allergens', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('preferred_portion_size', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('max_prep_time_minutes', sa.Integer(), nullable=True))
    op.add_column('users', sa.Column('max_cook_time_minutes', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove user preference fields
    op.drop_column('users', 'max_cook_time_minutes')
    op.drop_column('users', 'max_prep_time_minutes')
    op.drop_column('users', 'preferred_portion_size')
    op.drop_column('users', 'allergens')
    op.drop_column('users', 'dietary_restrictions')
