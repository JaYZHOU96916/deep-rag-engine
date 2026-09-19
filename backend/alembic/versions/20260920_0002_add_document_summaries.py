"""Add optional evidence-backed bilingual summaries.

Revision ID: 20260920_0002
Revises: 20260918_0001
Create Date: 2026-09-20 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260920_0002"
down_revision: str | None = "20260918_0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    summary_status = postgresql.ENUM(
        "pending",
        "summarizing",
        "completed",
        "failed",
        name="summary_status",
        create_type=False,
    )
    summary_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "document_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", summary_status, nullable=False, server_default="pending"),
        sa.Column("summary_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("source_chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("route_provider", sa.String(length=32), nullable=False),
        sa.Column("route_model", sa.String(length=200), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_document_summaries_status", "document_summaries", ["status"])
    op.create_index("ix_document_summaries_task_id", "document_summaries", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_document_summaries_task_id", table_name="document_summaries")
    op.drop_index("ix_document_summaries_status", table_name="document_summaries")
    op.drop_table("document_summaries")
    postgresql.ENUM(name="summary_status").drop(op.get_bind(), checkfirst=True)
