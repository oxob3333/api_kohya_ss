import os
import shutil
import json
from typing import List
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Importa tus módulos locales
from .database import engine, Base, get_db
from .models import (
    DBUser,
    UserCreate,
    UserResponse,
    Token,
    DBTrainingTask,
    # TrainingTaskCreate, # Ya no es necesario si se usan Forms
    TrainingTaskResponse,
)
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
)
from .tasks import train_kohya_model, UPLOADED_IMAGES_DIR, TRAINED_MODELS_DIR


app = FastAPI(title="Kohya_SS Training API")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    print("Database tables created and directories ensured.")


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    if user.email:
        db_user_email = db.query(DBUser).filter(DBUser.email == user.email).first()
        if db_user_email:
            raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        username=user.username, hashed_password=hashed_password, email=user.email
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


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
    # --- Parámetros del Formulario ---
    model_type: str = Form("lora"),
    prompt: str = Form("a photo of sks style"),
    instance_count: int = Form(10),
    class_name: str = Form("concept"),
    num_epochs: int = Form(10),
    learning_rate: float = Form(1e-5),
    resolution: int = Form(512),
    batch_size: int = Form(1),
    mixed_precision: str = Form("fp16"),
    use_xformers: bool = Form(True),
    enable_bucket: bool = Form(True),
    seed: int = Form(-1),
    output_name: str = Form("my_trained_model"),
    cache_latents: bool = Form(True),
    bucket_no_upscale: bool = Form(True),
    lr_scheduler: str = Form("cosine"),
    files: List[UploadFile] = File(...),
    # --- Dependencias ---
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="No images provided for training.")

    # 1. Crea una entrada preliminar en la BD para obtener un ID numérico.
    db_training_task = DBTrainingTask(
        user_id=current_user.id,
        status="initializing",  # Un estado temporal
        model_type=model_type,
    )
    db.add(db_training_task)
    db.commit()
    db.refresh(db_training_task)

    # Ahora tenemos un ID numérico: db_training_task.id
    task_id_from_db = db_training_task.id

    # 2. Usa el ID de la base de datos para construir la ruta de la tarea.
    user_task_base_dir = os.path.join(
        UPLOADED_IMAGES_DIR, f"user_{current_user.id}", f"task_{task_id_from_db}"
    )
    os.makedirs(user_task_base_dir, exist_ok=True)

    # 3. Guarda las imágenes en el directorio recién creado.
    saved_filenames = []
    for file in files:
        # Sanitizar el nombre del archivo es una buena práctica
        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(user_task_base_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_filenames.append(safe_filename)
        print(f"Saved {safe_filename} to {user_task_base_dir}")

    # 4. Agrupa todos los parámetros de entrenamiento para la tarea Celery.
    training_params = {
        "model_type": model_type,
        "prompt": prompt,
        "instance_count": instance_count,
        "class_name": class_name,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "cache_latents": cache_latents,
        "bucket_no_upscale": bucket_no_upscale,
        "lr_scheduler": lr_scheduler,
        "resolution": resolution,
        "batch_size": batch_size,
        "mixed_precision": mixed_precision,
        "use_xformers": use_xformers,
        "seed": seed,
        "enable_bucket": enable_bucket,
        "output_name": output_name,
    }

    # 5. Actualiza el registro de la tarea en la BD con la información final.
    db_training_task.status = "pending"  # Actualiza el estado a "pendiente"
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

    # 6. Despacha la tarea a Celery usando el ID numérico de la BD.
    print(f"Dispatching training task {db_training_task.id} to Celery...")
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
    tasks = (
        db.query(DBTrainingTask)
        .filter(DBTrainingTask.user_id == current_user.id)
        .order_by(DBTrainingTask.created_at.desc())
        .all()
    )
    return tasks


@app.get("/training-tasks/{task_id}", response_model=TrainingTaskResponse)
def get_training_task_status(
    task_id: int,  # El ID es un entero, lo que ahora es consistente
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
    task_id: int,  # El ID es un entero
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
            status_code=400,
            detail="Model is not yet available for download or training failed.",
        )

    if not os.path.exists(task.output_model_path) or not os.path.isfile(
        task.output_model_path
    ):
        raise HTTPException(status_code=500, detail="Model file not found on server.")

    filename = os.path.basename(task.output_model_path)
    return FileResponse(
        path=task.output_model_path,
        filename=filename,
        media_type="application/octet-stream",
    )
