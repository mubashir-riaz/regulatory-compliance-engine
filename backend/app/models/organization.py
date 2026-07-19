import uuid
from datetime import datetime
from typing import List

from sqlalchemy import String, DateTime, func, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
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
    evidence_artifacts: Mapped[List["EvidenceArtifact"]] = relationship(
        "EvidenceArtifact", back_populates="organization", cascade="all, delete-orphan"
    )
    compliance_mappings: Mapped[List["ComplianceMapping"]] = relationship(
        "ComplianceMapping", back_populates="organization", cascade="all, delete-orphan"
    )
