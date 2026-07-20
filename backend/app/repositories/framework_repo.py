from uuid import UUID
from typing import List, Optional
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.framework import RegulatoryFramework, RegulatoryVersion, RegulatoryRequirement
from app.repositories.base import BaseRepository

class RegulatoryFrameworkRepository(BaseRepository[RegulatoryFramework]):
    def __init__(self, session: AsyncSession):
        super().__init__(RegulatoryFramework, session)

    async def get_by_name(self, name: str) -> Optional[RegulatoryFramework]:
        result = await self.session.execute(
            select(self.model).filter(self.model.name == name)
        )
        return result.scalars().first()


class RegulatoryVersionRepository(BaseRepository[RegulatoryVersion]):
    def __init__(self, session: AsyncSession):
        super().__init__(RegulatoryVersion, session)

    async def get_by_slug(self, framework_id: UUID, version_slug: str) -> Optional[RegulatoryVersion]:
        result = await self.session.execute(
            select(self.model).filter(
                self.model.framework_id == framework_id,
                self.model.version_slug == version_slug
            )
        )
        return result.scalars().first()

    async def list_by_framework(self, framework_id: UUID) -> List[RegulatoryVersion]:
        result = await self.session.execute(
            select(self.model).filter(self.model.framework_id == framework_id)
        )
        return list(result.scalars().all())


class RegulatoryRequirementRepository(BaseRepository[RegulatoryRequirement]):
    def __init__(self, session: AsyncSession):
        super().__init__(RegulatoryRequirement, session)

    async def get_by_code(self, version_id: UUID, code: str) -> Optional[RegulatoryRequirement]:
        result = await self.session.execute(
            select(self.model).filter(
                self.model.version_id == version_id,
                self.model.code == code
            )
        )
        return result.scalars().first()

    async def list_by_version(self, version_id: UUID) -> List[RegulatoryRequirement]:
        result = await self.session.execute(
            select(self.model).filter(self.model.version_id == version_id)
        )
        return list(result.scalars().all())
