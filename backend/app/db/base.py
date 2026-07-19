from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Import all models here so Alembic can discover them
from app.models.organization import Organization  # noqa: F401
from app.models.framework import RegulatoryFramework, RegulatoryVersion, RegulatoryRequirement  # noqa: F401
from app.models.evidence import EvidenceArtifact  # noqa: F401
from app.models.compliance_mapping import ComplianceMapping  # noqa: F401