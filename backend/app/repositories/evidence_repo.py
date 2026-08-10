from uuid import UUID
from typing import List, Optional
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evidence import EvidenceArtifact
from app.models.compliance_mapping import ComplianceMapping
from app.repositories.base import BaseRepository

class EvidenceArtifactRepository(BaseRepository[EvidenceArtifact]):
    def __init__(self, session: AsyncSession):
        super().__init__(EvidenceArtifact, session)

    async def list_by_organization(
        self, organization_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[EvidenceArtifact]:
        result = await self.session.execute(
            select(self.model)
            .filter(self.model.organization_id == organization_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_file_path(self, organization_id: UUID, file_path: str) -> Optional[EvidenceArtifact]:
        result = await self.session.execute(
            select(self.model).filter(
                self.model.organization_id == organization_id,
                self.model.file_path == file_path
            )
        )
        return result.scalars().first()


class ComplianceMappingRepository(BaseRepository[ComplianceMapping]):
    def __init__(self, session: AsyncSession):
        super().__init__(ComplianceMapping, session)

    async def list_by_organization(
        self, organization_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[ComplianceMapping]:
        result = await self.session.execute(
            select(self.model)
            .filter(self.model.organization_id == organization_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_evidence(self, evidence_id: UUID) -> List[ComplianceMapping]:
        result = await self.session.execute(
            select(self.model).filter(self.model.evidence_id == evidence_id)
        )
        return list(result.scalars().all())

    async def list_by_requirement(self, requirement_id: UUID) -> List[ComplianceMapping]:
        result = await self.session.execute(
            select(self.model).filter(self.model.requirement_id == requirement_id)
        )
        return list(result.scalars().all())
