from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, ConfigDict, Field
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
    celery_task_id = Column(
        String, nullable=True, index=True
    )  # NUEVO: ID de la tarea Celery
    status = Column(
        String, default="pending"
    )  # pending, initializing, processing, completed, failed
    model_type = Column(String)
    dataset_path = Column(String)  # Ruta base donde se extraen las imágenes
    output_model_path = Column(String, nullable=True)  # Ruta al modelo LoRA final
    prompt_config = Column(
        String, nullable=True
    )  # JSON string de la configuración de prompt
    error_message = Column(Text, nullable=True)  # NUEVO: Para guardar mensajes de error
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


# TrainingTaskCreate ya no se usa directamente si todos los params son Form en el endpoint
# class TrainingTaskCreate(BaseModel):
#     # ... (podrías mantenerlo si tienes una estructura de datos interna)


class TrainingTaskResponse(BaseModel):
    id: int
    user_id: int
    celery_task_id: Optional[str] = None
    status: str
    model_type: str
    dataset_path: str
    output_model_path: Optional[str] = None
    prompt_config: Optional[str] = None
    error_message: Optional[str] = None

    # Campos para el progreso en tiempo real
    progress_text: Optional[str] = Field(
        default=None, description="Última línea de log relevante o mensaje de progreso."
    )
    progress_percent: Optional[float] = Field(
        default=None, description="Porcentaje de progreso estimado."
    )

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
