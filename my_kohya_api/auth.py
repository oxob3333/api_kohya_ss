import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt  # Asegúrate de que sea python-jose, no el jose antiguo
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .database import get_db  # Suponiendo que get_db está en database.py
from .models import DBUser  # Suponiendo que DBUser está en models.py

# --- Configuración de Seguridad ---
# ¡CAMBIA ESTO EN PRODUCCIÓN! Usa 'openssl rand -hex 32' para generar una clave segura
SECRET_KEY = os.environ.get(
    "SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Para el hashing de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Para la obtención del token Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Credenciales del Usuario Maestro (para la dependencia de verificación) ---
# Leemos de app.py o lo definimos aquí. Para evitar importación circular si app.py importa de auth.py,
# es más simple definirlo aquí o usar una variable de entorno.
# Por ahora, lo definiremos aquí para que este archivo sea autocontenido para la prueba.
# Idealmente, esto vendría de una configuración central o variables de entorno.
APP_MASTER_USERNAME = os.environ.get("MASTER_USER_USERNAME", "dorian")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: DBUser = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# --- NUEVA DEPENDENCIA PARA PROTEGER EL REGISTRO ---
async def get_current_master_user(  # Renombrada para claridad, puedes llamarla como quieras
    current_user: DBUser = Depends(
        get_current_active_user
    ),  # Reutiliza la dependencia de usuario activo
):
    # APP_MASTER_USERNAME es el nombre de usuario definido para el maestro
    # (ej. "dorian" o lo que esté en tus variables de entorno para MASTER_USER_USERNAME)
    if current_user.username != APP_MASTER_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para registrar nuevos usuarios.",
        )
    return current_user
