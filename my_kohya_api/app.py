import os
import shutil
import json
import zipfile
import io
from typing import List, Optional  # Asegúrate de que Optional esté aquí
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Importa tus módulos locales
from .database import engine, Base, SessionLocal, get_db
from .models import (
    DBUser,
    UserCreate,
    UserResponse,
    Token,
    DBTrainingTask,
    TrainingTaskResponse,
)
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_master_user,
)
from .tasks import (
    train_kohya_model,
    UPLOADED_IMAGES_DIR,
    TRAINED_MODELS_DIR,
    celery_app,
)  # Importar celery_app


# --- Credenciales del Usuario Maestro ---
MASTER_USERNAME = os.environ.get("MASTER_USER_USERNAME", "dorian")
MASTER_PASSWORD = os.environ.get("MASTER_USER_PASSWORD", "doritos")
MASTER_EMAIL = os.environ.get("MASTER_USER_EMAIL", "dorian_ain@hotmail.com")

app = FastAPI(title="Kohya_SS Training API")


@app.on_event("startup")
def on_startup():
    print("Ejecutando evento de inicio...")
    Base.metadata.create_all(bind=engine)  # Esto crea las tablas si no existen
    print("Tablas de base de datos verificadas/creadas.")
    os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    print("Directorios de datos asegurados.")

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


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(
    user_to_create: UserCreate,
    db: Session = Depends(get_db),
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
        is_active=True,
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
    model_type: str = Form("lora"),
    prompt: str = Form("a photo of sks style"),
    instance_count: int = Form(10),
    class_name: str = Form("concept"),
    num_epochs: int = Form(5),
    learning_rate: float = Form(1e-5),
    resolution: int = Form(512),
    train_batch_size: int = Form(1),
    mixed_precision: str = Form("fp16"),
    use_xformers: bool = Form(True),
    enable_bucket: bool = Form(True),
    seed: int = Form(-1),
    output_name: str = Form("my_trained_model"),
    cache_latents: bool = Form(True),
    bucket_no_upscale: bool = Form(True),
    lr_scheduler: str = Form("cosine"),
    network_dim: int = Form(128),
    network_alpha: int = Form(64),
    optimizer_type: str = Form("AdamW8bit"),
    zip_file: UploadFile = File(...),
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

    celery_task_obj = train_kohya_model.delay(
        db_training_task.id,
        current_user.id,
        user_task_base_dir,
        model_type,
        training_params,
        saved_filenames,
    )
    db_training_task.celery_task_id = celery_task_obj.id  # Guardar ID de Celery

    db.add(db_training_task)
    db.commit()
    db.refresh(db_training_task)
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
    db_task = (
        db.query(DBTrainingTask)
        .filter(DBTrainingTask.id == task_id, DBTrainingTask.user_id == current_user.id)
        .first()
    )
    if not db_task:
        raise HTTPException(
            status_code=404, detail="Training task not found or unauthorized."
        )

    current_celery_status = db_task.status
    progress_text_from_celery = None
    progress_percent_from_celery = None
    error_message_from_celery = db_task.error_message  # Usar el de la BD como fallback

    if db_task.celery_task_id:
        async_result = celery_app.AsyncResult(db_task.celery_task_id)
        current_celery_status = async_result.state

        if async_result.state == "PROGRESS":
            if isinstance(async_result.info, dict):
                progress_text_from_celery = async_result.info.get("progress_text")
                progress_percent_from_celery = async_result.info.get("progress_percent")
        elif async_result.state == "FAILURE":
            error_message_from_celery = str(async_result.info)
            if (
                db_task.status != "failed"
                or db_task.error_message != error_message_from_celery
            ):
                db_task.status = "failed"
                db_task.error_message = error_message_from_celery
                db.commit()
        elif async_result.state == "SUCCESS":
            if db_task.status != "completed":
                db_task.status = "completed"  # Marcar como completado si Celery lo dice y BD no lo está
                db_task.error_message = (
                    None  # Limpiar errores previos si ahora es SUCCESS
                )
                db.commit()

    # Si el estado de la BD es 'processing' pero Celery no tiene info de progreso detallada,
    # o si la tarea aún no ha sido recogida por Celery (estado Celery es PENDING).
    if current_celery_status == "PROCESSING" and not progress_text_from_celery:
        progress_text_from_celery = "Procesando, esperando detalles de Kohya..."
    elif current_celery_status == "PENDING" and db_task.status == "pending":
        progress_text_from_celery = "Tarea pendiente en la cola de Celery..."

    # Construir la respuesta Pydantic
    return TrainingTaskResponse(
        id=db_task.id,
        user_id=db_task.user_id,
        celery_task_id=db_task.celery_task_id,
        status=current_celery_status,  # Priorizar estado de Celery
        model_type=db_task.model_type,
        dataset_path=db_task.dataset_path,
        output_model_path=db_task.output_model_path,
        prompt_config=db_task.prompt_config,
        error_message=error_message_from_celery,
        progress_text=progress_text_from_celery,
        progress_percent=progress_percent_from_celery,
        created_at=db_task.created_at,
        updated_at=db_task.updated_at,  # La BD actualiza esto, o Celery podría si es necesario
    )


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

    # Consultar el estado real de Celery aquí también podría ser útil antes de permitir la descarga
    celery_status_for_download = task.status
    if task.celery_task_id:
        celery_status_for_download = celery_app.AsyncResult(task.celery_task_id).state

    if (
        celery_status_for_download != "SUCCESS" or not task.output_model_path
    ):  # Chequear contra SUCCESS de Celery
        raise HTTPException(
            status_code=400,
            detail="Model not available or training did not complete successfully.",
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
