import os
import shutil
import json
import subprocess
from celery import Celery
from sqlalchemy.orm import Session
from datetime import datetime
import sys
import random
from typing import List

# Importa desde tus módulos locales
from .database import SessionLocal, Base, engine  # Asegúrate que SessionLocal esté aquí
from .models import DBTrainingTask

# --- Configuración de Celery ---
celery_app = Celery(
    "my_kohya_api_tasks",
    broker="amqp://guest:guest@localhost:5672//",
    backend="db+sqlite:///./celery_results.db",
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

# --- Rutas de Archivos ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOADED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "uploaded_images")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")


# --- Tarea de Entrenamiento de Kohya_SS ---
@celery_app.task(bind=True)
def train_kohya_model(
    self,
    task_id: int,
    user_id: int,
    user_task_base_dir: str,
    model_type: str,
    training_params: dict,
    saved_filenames: List[str],
):
    db = SessionLocal()  # Abre una nueva sesión de BD para esta ejecución de tarea.
    training_task = None  # Inicializa para asegurar que siempre esté definido en el bloque 'finally'.

    try:
        # Recupera la tarea de la base de datos para actualizar su estado.
        training_task = (
            db.query(DBTrainingTask).filter(DBTrainingTask.id == task_id).first()
        )
        if not training_task:
            raise Exception(
                f"Tarea de entrenamiento con ID {task_id} no encontrada en la base de datos."
            )

        # Actualiza el estado de la tarea a 'processing' (en proceso).
        training_task.status = "processing"
        db.add(training_task)
        db.commit()
        db.refresh(training_task)

        print(
            f"[{datetime.now()}] Iniciando entrenamiento para la tarea {task_id} (Usuario: {user_id})"
        )

        # --- Obtener parámetros específicos para la estructura de carpetas de Kohya_SS ---
        instance_count = training_params.get("instance_count", 10)
        class_name = training_params.get("class_name", "my_concept")
        user_prompt = training_params.get("prompt", "default prompt if none provided")
        output_name_from_user = training_params.get(
            "output_name", f"model_lora_{task_id}"
        )

        # --- Crear la subcarpeta [instance_count]_[class_name] de Kohya_SS ---
        kohya_dataset_dir = os.path.join(
            user_task_base_dir, f"{instance_count}_{class_name}"
        )
        os.makedirs(kohya_dataset_dir, exist_ok=True)
        print(f"[{datetime.now()}] Creando subcarpeta de Kohya_SS: {kohya_dataset_dir}")

        # --- Mover imágenes y generar los archivos .txt ---
        print(
            f"\n--- Movimiento de imágenes y generación de captions con el prompt: '{user_prompt}' ---"
        )
        for filename in saved_filenames:
            original_file_path = os.path.join(user_task_base_dir, filename)
            kohya_file_path = os.path.join(kohya_dataset_dir, filename)

            if os.path.exists(original_file_path):
                shutil.move(original_file_path, kohya_file_path)
            else:
                # Si el archivo ya fue movido o no existe, solo crea el caption
                print(
                    f"   Advertencia: Archivo {original_file_path} no encontrado para mover. Saltando movimiento."
                )

            base_name = os.path.splitext(filename)[0]
            txt_filename = base_name + ".txt"
            txt_filepath = os.path.join(kohya_dataset_dir, txt_filename)
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.write(user_prompt)
            print(f"   Creado/Actualizado caption para: {filename}")
        print("--- Movimiento y captions completados. ---")

        # --- Manejo del Seed ---
        actual_seed = training_params.get("seed", -1)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)
            print(f"[{datetime.now()}] Generando semilla aleatoria: {actual_seed}")
        else:
            print(f"[{datetime.now()}] Usando semilla proporcionada: {actual_seed}")

        # --- Preparar la Carpeta de Salida del Modelo ---
        output_dir = os.path.join(
            TRAINED_MODELS_DIR, f"user_{user_id}", f"task_{task_id}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_model_path_full = os.path.join(
            output_dir, f"{output_name_from_user}.safetensors"
        )

        # --- Invocar el Script de Entrenamiento de Kohya_SS ---
        kohya_script_wrapper_path = os.path.join(
            PROJECT_ROOT, "my_kohya_api", "kohya_scripts", "train_lora.py"
        )

        # --- Construcción del Comando ---
        command = [
            sys.executable,
            kohya_script_wrapper_path,
            "--pretrained_model_name_or_path",
            "runwayml/stable-diffusion-v1-5",
            "--train_data_dir",
            user_task_base_dir,  # <-- CORREGIDO
            "--output_dir",
            output_dir,
            "--output_name",
            output_name_from_user,
            "--resolution",
            str(training_params.get("resolution", 512)),
            "--train_batch_size",
            str(training_params.get("batch_size", 1)),
            "--num_epochs",
            str(training_params.get("num_epochs", 10)),
            "--learning_rate",
            str(training_params.get("learning_rate", 1e-4)),
            "--mixed_precision",
            training_params.get("mixed_precision", "fp16"),
            "--seed",
            str(actual_seed),
            "--network_dim",
            "128",
            "--network_alpha",
            "64",
            "--optimizer_type",
            training_params.get("optimizer_type", "AdamW8bit"),
            "--caption_extension",
            ".txt",
            "--lr_scheduler",
            training_params.get("lr_scheduler", "constant"),
        ]

        if training_params.get("use_xformers", True):
            command.append("--xformers")
        if training_params.get("enable_bucket", False):
            command.append("--enable_bucket")
        if training_params.get("cache_latents"):
            command.append("--cache_latents")
        if training_params.get("bucket_no_upscale"):
            command.append("--bucket_no_upscale")

        print(f"\n[{datetime.now()}] Ejecutando comando: {' '.join(command)}")

        # --- MODIFICACIÓN PARA STREAMING EN TIEMPO REAL ---
        print(f"\n[{datetime.now()}] Iniciando streaming del proceso Kohya_SS...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(kohya_script_wrapper_path),
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
            process.stdout.close()

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output="El error se puede ver en el log de Celery.",
            )

        print(
            f"\n[{datetime.now()}] Proceso de Kohya_SS finalizado con código {return_code}."
        )
        # --- FIN DE LA MODIFICACIÓN ---

        # --- Verificar y Finalizar Tarea ---
        if (
            os.path.exists(output_model_path_full)
            and os.path.getsize(output_model_path_full) > 0
        ):
            training_task.status = "completed"
            training_task.output_model_path = output_model_path_full
            print(
                f"[{datetime.now()}] Tarea {task_id} completada. Modelo: {output_model_path_full}"
            )
        else:
            raise Exception(
                f"El archivo del modelo no fue encontrado o está vacío en {output_model_path_full}"
            )

        training_task.updated_at = datetime.utcnow()
        db.add(training_task)
        db.commit()
        db.refresh(training_task)

    except subprocess.CalledProcessError as e:
        print(
            f"[{datetime.now()}] El subproceso de Kohya_SS falló para la tarea {task_id} con código {e.returncode}.",
            file=sys.stderr,
        )
        if training_task:
            training_task.status = "failed"
            training_task.updated_at = datetime.utcnow()
            db.add(training_task)
            db.commit()
        raise

    except Exception as e:
        print(
            f"[{datetime.now()}] Error inesperado en la tarea {task_id}: {e}",
            file=sys.stderr,
        )
        if training_task:
            training_task.status = "failed"
            training_task.updated_at = datetime.utcnow()
            db.add(training_task)
            db.commit()
        raise

    finally:
        if db:
            db.close()
