from uuid import UUID
from typing import Optional
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import Organization
from app.repositories.base import BaseRepository

class OrganizationRepository(BaseRepository[Organization]):
    def __init__(self, session: AsyncSession):
        super().__init__(Organization, session)

    async def get_by_name(self, name: str) -> Optional[Organization]:
        result = await self.session.execute(
            select(self.model).filter(self.model.name == name)
        )
        return result.scalars().first()
