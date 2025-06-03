from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Usaremos SQLite por simplicidad. Para MySQL, cambiar a 'mysql+pymysql://user:pass@host:port/dbname'
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Para MySQL, necesitarías:
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://user:password@host/db_name"
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # Solo para SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependencia para obtener una sesión de BD en FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
