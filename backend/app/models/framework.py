from __future__ import annotations
import uuid
from datetime import datetime, date
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import String, Text, DateTime, Date, Boolean, ForeignKey, UniqueConstraint, func, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.compliance_mapping import ComplianceMapping

class RegulatoryFramework(Base):
    __tablename__ = "regulatory_frameworks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
    versions: Mapped[List["RegulatoryVersion"]] = relationship(
        "RegulatoryVersion", back_populates="framework", cascade="all, delete-orphan"
    )


class RegulatoryVersion(Base):
    __tablename__ = "regulatory_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    framework_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulatory_frameworks.id", ondelete="CASCADE"), nullable=False
    )
    version_slug: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "v1", "2016", "2022"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
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
    framework: Mapped["RegulatoryFramework"] = relationship("RegulatoryFramework", back_populates="versions")
    requirements: Mapped[List["RegulatoryRequirement"]] = relationship(
        "RegulatoryRequirement", back_populates="version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("framework_id", "version_slug", name="uq_framework_version"),
    )


class RegulatoryRequirement(Base):
    __tablename__ = "regulatory_requirements"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulatory_versions.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "Art. 32", "CC1.1"
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
    version: Mapped["RegulatoryVersion"] = relationship("RegulatoryVersion", back_populates="requirements")
    compliance_mappings: Mapped[List["ComplianceMapping"]] = relationship(
        "ComplianceMapping", back_populates="requirement", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("version_id", "code", name="uq_version_requirement_code"),
    )
