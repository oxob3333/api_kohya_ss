from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

from .database import Base

# --- Modelos de Base de Datos (SQLAlchemy) ---


class DBUser(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True)

    training_tasks = relationship("DBTrainingTask", back_populates="owner")


class DBTrainingTask(Base):
    __tablename__ = "training_tasks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="pending")
    model_type = Column(String)
    dataset_path = Column(String)
    output_model_path = Column(String, nullable=True)
    prompt_config = Column(
        String, nullable=True
    )  # JSON string of prompt configurations
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    owner = relationship("DBUser", back_populates="training_tasks")


# --- Esquemas Pydantic ---


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str


class TrainingTaskCreate(BaseModel):
    model_type: str = "lora"
    prompt: str = "default prompt for training"
    instance_count: int = 10  # <--- NUEVO: Cuenta de instancias para Kohya_SS
    class_name: str = (
        "my_concept"  # <--- NUEVO: Nombre de la clase para Kohya_SS (ej. "person", "style")
    )
    num_epochs: int = 10
    learning_rate: float = 1e-4
    resolution: int = 512
    batch_size: int = 1
    mixed_precision: str = "fp16"
    use_xformers: bool = True
    seed: int = -1


class TrainingTaskResponse(BaseModel):
    id: int
    user_id: int
    status: str
    model_type: str
    dataset_path: str
    output_model_path: Optional[str] = None
    prompt_config: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
