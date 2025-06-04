import os
import shutil
import json
import zipfile
import io
from typing import List
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Importa tus módulos locales
from .database import (
    engine,
    Base,
    SessionLocal,
    get_db,
)  # Asegúrate de importar SessionLocal y get_db
from .models import (
    DBUser,
    UserCreate,
    UserResponse,
    Token,
    DBTrainingTask,
    TrainingTaskResponse,
)
from .auth import (  # Asegúrate de que las importaciones de auth sean correctas
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    # ¡NUEVA DEPENDENCIA QUE AÑADIREMOS EN auth.py!
    get_current_master_user,  # Renombrada para claridad, la crearemos en auth.py
)
from .tasks import train_kohya_model, UPLOADED_IMAGES_DIR, TRAINED_MODELS_DIR


# --- Credenciales del Usuario Maestro ---
# ¡¡¡IMPORTANTE!!! Para producción, considera seriamente leer esto desde variables de entorno.
# Por ahora, usamos tus valores especificados como default si no hay variables de entorno.
MASTER_USERNAME = os.environ.get("MASTER_USER_USERNAME", "dorian")
MASTER_PASSWORD = os.environ.get("MASTER_USER_PASSWORD", "doritos")
MASTER_EMAIL = os.environ.get("MASTER_USER_EMAIL", "dorian_ain@hotmail.com")


app = FastAPI(title="Kohya_SS Training API")


@app.on_event("startup")
def on_startup():
    print("Ejecutando evento de inicio...")
    Base.metadata.create_all(bind=engine)
    print("Tablas de base de datos verificadas/creadas.")
    os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    print("Directorios de datos asegurados.")

    # --- Crear Usuario Maestro si no existe ---
    db: Session = SessionLocal()
    try:
        master_user = (
            db.query(DBUser).filter(DBUser.username == MASTER_USERNAME).first()
        )
        if not master_user:
            print(f"Usuario maestro '{MASTER_USERNAME}' no encontrado, creándolo...")
            hashed_password = get_password_hash(MASTER_PASSWORD)
            db_master_user = DBUser(
                username=MASTER_USERNAME,
                hashed_password=hashed_password,
                email=MASTER_EMAIL,
                is_active=True,
                # Aquí podrías añadir un campo 'is_superuser=True' a tu modelo DBUser
                # y establecerlo aquí para una verificación más formal en la dependencia.
            )
            db.add(db_master_user)
            db.commit()
            db.refresh(db_master_user)
            print(
                f"Usuario maestro '{MASTER_USERNAME}' creado exitosamente con email '{MASTER_EMAIL}'."
            )
        else:
            print(f"Usuario maestro '{MASTER_USERNAME}' ya existe.")
    finally:
        db.close()
    print("Evento de inicio completado.")


# --- Endpoint de Registro AHORA PROTEGIDO ---
@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(
    user_to_create: UserCreate,  # Los datos del nuevo usuario a crear
    db: Session = Depends(get_db),
    # ¡Esta dependencia asegura que solo el maestro pueda usar este endpoint!
    current_master_user: DBUser = Depends(get_current_master_user),
):
    print(
        f"Intento de registro por: '{current_master_user.username}'. Creando usuario: '{user_to_create.username}'"
    )
    db_user = (
        db.query(DBUser).filter(DBUser.username == user_to_create.username).first()
    )
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    if user_to_create.email:
        db_user_email = (
            db.query(DBUser).filter(DBUser.email == user_to_create.email).first()
        )
        if db_user_email:
            raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user_to_create.password)
    new_db_user = DBUser(
        username=user_to_create.username,
        hashed_password=hashed_password,
        email=user_to_create.email,
        is_active=True,  # Por defecto, los usuarios creados por el admin están activos
    )
    db.add(new_db_user)
    db.commit()
    db.refresh(new_db_user)
    return new_db_user


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: DBUser = Depends(get_current_user)):
    return current_user


@app.post(
    "/training-tasks",
    response_model=TrainingTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_training_task(
    # Parámetros de Formulario para Kohya_SS
    model_type: str = Form("lora", description="Tipo de modelo a entrenar, ej. 'lora'"),
    prompt: str = Form(
        "a photo of sks style",
        description="Prompt base para las imágenes de instancia.",
    ),
    instance_count: int = Form(
        10,
        description="Número de repeticiones por imagen de instancia (ej. 10_nombreclase).",
    ),
    class_name: str = Form(
        "concept",
        description="Nombre de la clase para la carpeta (ej. 10_nombreclase).",
    ),
    num_epochs: int = Form(5, description="Número de épocas de entrenamiento."),
    learning_rate: float = Form(1e-5, description="Tasa de aprendizaje."),
    resolution: int = Form(512, description="Resolución de entrenamiento (cuadrada)."),
    train_batch_size: int = Form(
        1, description="Tamaño del lote de entrenamiento por dispositivo."
    ),
    mixed_precision: str = Form(
        "fp16", description="Precisión mixta (no, fp16, bf16)."
    ),
    use_xformers: bool = Form(True, description="Usar xformers para optimización."),
    enable_bucket: bool = Form(True, description="Habilitar bucketing."),
    seed: int = Form(
        -1, description="Semilla para reproducibilidad (-1 para aleatorio)."
    ),
    output_name: str = Form(
        "my_trained_model",
        description="Nombre base para el archivo del modelo LoRA resultante.",
    ),
    cache_latents: bool = Form(
        True, description="Cachear latentes para acelerar el entrenamiento."
    ),
    bucket_no_upscale: bool = Form(
        True,
        description="No agrandar imágenes en buckets, usar su resolución más cercana.",
    ),
    lr_scheduler: str = Form(
        "cosine", description="Planificador de tasa de aprendizaje."
    ),
    network_dim: int = Form(128, description="Dimensión de la red LoRA (rank)."),
    network_alpha: int = Form(64, description="Alpha para la red LoRA."),
    optimizer_type: str = Form("AdamW8bit", description="Optimizador a usar."),
    zip_file: UploadFile = File(
        ..., description="Archivo .zip conteniendo las imágenes de entrenamiento."
    ),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db_training_task = DBTrainingTask(
        user_id=current_user.id,
        status="initializing",
        model_type=model_type,
    )
    db.add(db_training_task)
    db.commit()
    db.refresh(db_training_task)

    task_id_from_db = db_training_task.id
    user_task_base_dir = os.path.join(
        UPLOADED_IMAGES_DIR, f"user_{current_user.id}", f"task_{task_id_from_db}"
    )
    os.makedirs(user_task_base_dir, exist_ok=True)

    saved_filenames = []
    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    try:
        zip_content = await zip_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zf:
            for member_name in zf.namelist():
                if (
                    not member_name.startswith("/")
                    and ".." not in member_name
                    and not member_name.endswith("/")
                    and member_name.lower().endswith(image_extensions)
                ):
                    filename = os.path.basename(member_name)
                    if not filename:
                        continue
                    file_path = os.path.join(user_task_base_dir, filename)
                    source = zf.open(member_name)
                    target = open(file_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                    saved_filenames.append(filename)
    except zipfile.BadZipFile:
        db.delete(db_training_task)
        db.commit()
        raise HTTPException(status_code=400, detail="Invalid ZIP file provided.")
    except Exception as e:
        db.delete(db_training_task)
        db.commit()
        raise HTTPException(
            status_code=500, detail=f"Error processing ZIP file: {str(e)}"
        )

    if not saved_filenames:
        db.delete(db_training_task)
        db.commit()
        raise HTTPException(
            status_code=400, detail="No valid image files found in ZIP."
        )

    training_params = {
        "model_type": model_type,
        "prompt": prompt,
        "instance_count": instance_count,
        "class_name": class_name,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "train_batch_size": train_batch_size,
        "mixed_precision": mixed_precision,
        "use_xformers": use_xformers,
        "enable_bucket": enable_bucket,
        "seed": seed,
        "output_name": output_name,
        "cache_latents": cache_latents,
        "bucket_no_upscale": bucket_no_upscale,
        "lr_scheduler": lr_scheduler,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "optimizer_type": optimizer_type,
    }

    db_training_task.status = "pending"
    db_training_task.dataset_path = user_task_base_dir
    db_training_task.prompt_config = json.dumps(
        {
            "user_prompt": prompt,
            "instance_count": instance_count,
            "class_name": class_name,
            "output_name": output_name,
        }
    )
    db.add(db_training_task)
    db.commit()
    db.refresh(db_training_task)

    train_kohya_model.delay(
        db_training_task.id,
        current_user.id,
        user_task_base_dir,
        model_type,
        training_params,
        saved_filenames,
    )
    return db_training_task


@app.get("/training-tasks", response_model=List[TrainingTaskResponse])
def get_user_training_tasks(
    current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)
):
    return (
        db.query(DBTrainingTask)
        .filter(DBTrainingTask.user_id == current_user.id)
        .order_by(DBTrainingTask.created_at.desc())
        .all()
    )


@app.get("/training-tasks/{task_id}", response_model=TrainingTaskResponse)
def get_training_task_status(
    task_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = (
        db.query(DBTrainingTask)
        .filter(DBTrainingTask.id == task_id, DBTrainingTask.user_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=404, detail="Training task not found or unauthorized."
        )
    return task


@app.get("/download-model/{task_id}")
async def download_trained_model(
    task_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = (
        db.query(DBTrainingTask)
        .filter(DBTrainingTask.id == task_id, DBTrainingTask.user_id == current_user.id)
        .first()
    )
    if not task:
        raise HTTPException(
            status_code=404, detail="Training task not found or unauthorized."
        )
    if task.status != "completed" or not task.output_model_path:
        raise HTTPException(
            status_code=400, detail="Model not available or training failed."
        )
    if not os.path.exists(task.output_model_path) or not os.path.isfile(
        task.output_model_path
    ):
        raise HTTPException(status_code=500, detail="Model file not found on server.")
    return FileResponse(
        path=task.output_model_path,
        filename=os.path.basename(task.output_model_path),
        media_type="application/octet-stream",
    )
