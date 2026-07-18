from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Import all models here later so Alembic can discover them
# from app.models import user, tenant, ...