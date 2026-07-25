import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

class EvidenceArtifact(Base):
    __tablename__ = "evidence_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)  # original filename
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)  # MinIO path/key
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # parsed content
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), server_default="PENDING", nullable=False)  # PENDING, PROCESSING, COMPLETED, FAILED
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="evidence_artifacts")
    compliance_mappings: Mapped[List["ComplianceMapping"]] = relationship(
        "ComplianceMapping", back_populates="evidence", cascade="all, delete-orphan"
    )
