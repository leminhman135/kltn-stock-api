"""Database connection and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# If DATABASE_URL is not set, use SQLite (file-based database)
if not DATABASE_URL:
    # Use SQLite for simplicity - no server needed!
    DATABASE_URL = "sqlite:///./kltn_stocks.db"
    print(f"[INFO] Using SQLite database: kltn_stocks.db")
else:
    print(f"[INFO] Using database from env: {DATABASE_URL}")
    # Railway PostgreSQL URLs start with postgres:// but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        print(f"[INFO] Converted postgres:// to postgresql://")

# Create engine
# For SQLite, use StaticPool to avoid threading issues
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    # For PostgreSQL/other databases
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function for FastAPI to get database session.
    
    Usage in FastAPI:
        @app.get("/stocks")
        def get_stocks(db: Session = Depends(get_db)):
            return db.query(Stock).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables."""
    from .models import Base
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def drop_db():
    """Drop all tables - use with caution!"""
    from .models import Base
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped!")


if __name__ == "__main__":
    # Test connection
    try:
        with engine.connect() as conn:
            print("✅ Database connection successful!")
            print(f"📍 Connected to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'local'}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
