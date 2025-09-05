from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import os
from backend.config import settings

if settings.ENVIRONMENT == "production" and settings.DATABASE_URL_PROD:
    DATABASE_URL = settings.DATABASE_URL_PROD
else:
    DATABASE_URL = settings.DATABASE_URL

engine=create_engine(DATABASE_URL,
              pool_pre_ping=True,
              pool_size=3600,
              echo=settings.DEBUG
              )

SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)

Base=declarative_base()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

