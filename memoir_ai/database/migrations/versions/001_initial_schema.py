"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""

    # Create categories table
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("level", sa.SmallInteger(), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("slug", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("level >= 1 AND level <= 100", name="valid_level_range"),
        sa.CheckConstraint(
            "(level = 1 AND parent_id IS NULL) OR (level > 1 AND parent_id IS NOT NULL)",
            name="parent_level_consistency",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["categories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent_id", name="unique_name_per_parent"),
        sa.UniqueConstraint("slug"),
    )

    # Create indexes for categories
    op.create_index("idx_categories_level", "categories", ["level"])
    op.create_index("idx_categories_name", "categories", ["name"])
    op.create_index("idx_categories_parent_id", "categories", ["parent_id"])
    op.create_index(
        "uq_categories_root_name",
        "categories",
        ["name"],
        unique=True,
        sqlite_where=sa.text("parent_id IS NULL"),
        postgresql_where=sa.text("parent_id IS NULL"),
    )

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("token_count > 0", name="positive_token_count"),
        sa.CheckConstraint("length(content) > 0", name="non_empty_content"),
        sa.ForeignKeyConstraint(
            ["category_id"],
            ["categories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for chunks
    op.create_index("idx_chunks_category_id", "chunks", ["category_id"])
    op.create_index("idx_chunks_created_at", "chunks", ["created_at"])
    op.create_index("idx_chunks_source_id", "chunks", ["source_id"])
    op.create_index("idx_chunks_token_count", "chunks", ["token_count"])

    # Create contextual_helpers table
    op.create_table(
        "contextual_helpers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("helper_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("is_user_provided", sa.Boolean(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("token_count > 0", name="positive_helper_token_count"),
        sa.CheckConstraint("token_count <= 300", name="helper_token_limit"),
        sa.CheckConstraint("length(helper_text) > 0", name="non_empty_helper_text"),
        sa.CheckConstraint("version > 0", name="positive_version"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )

    # Create indexes for contextual_helpers
    op.create_index(
        "idx_contextual_helpers_created_at", "contextual_helpers", ["created_at"]
    )
    op.create_index(
        "idx_contextual_helpers_source_id", "contextual_helpers", ["source_id"]
    )

    # Create category_limits table
    op.create_table(
        "category_limits",
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("max_categories", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "level >= 1 AND level <= 100", name="valid_limit_level_range"
        ),
        sa.CheckConstraint("max_categories > 0", name="positive_max_categories"),
        sa.PrimaryKeyConstraint("level"),
    )


def downgrade() -> None:
    """Drop all tables."""

    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table("category_limits")
    op.drop_table("contextual_helpers")
    op.drop_table("chunks")
    op.drop_index("uq_categories_root_name", table_name="categories")
    op.drop_table("categories")
