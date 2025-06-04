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
from .database import SessionLocal  # Asegúrate que SessionLocal esté aquí
from .models import DBTrainingTask

# --- Configuración de Celery ---
celery_app = Celery(
    "my_kohya_api_tasks",
    broker="amqp://guest:guest@localhost:5672//",  # O la URL de tu RabbitMQ si es diferente
    backend="db+sqlite:///./celery_results.db",
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,  # Crucial para tareas de GPU, procesar una a la vez por worker
    broker_connection_retry_on_startup=True,
)

# --- Rutas de Archivos ---
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Ajustado para estar en la raíz del proyecto si api_kohya_ss es un subdirectorio
UPLOADED_IMAGES_DIR = os.path.join(
    PROJECT_ROOT, "api_kohya_ss", "uploaded_images"
)  # Asumiendo que api_kohya_ss es el nombre de la app
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "api_kohya_ss", "trained_models")
KOHYA_SCRIPTS_DIR = os.path.join(
    PROJECT_ROOT, "api_kohya_ss", "my_kohya_api", "kohya_scripts"
)


@celery_app.task(bind=True)
def train_kohya_model(
    self,
    task_id: int,
    user_id: int,
    user_task_base_dir: str,  # Directorio donde se extrajeron las imágenes
    model_type: str,
    training_params: dict,
    saved_filenames: List[str],  # Nombres de los archivos extraídos del ZIP
):
    db = SessionLocal()
    training_task = None

    try:
        training_task = (
            db.query(DBTrainingTask).filter(DBTrainingTask.id == task_id).first()
        )
        if not training_task:
            raise Exception(f"Tarea de entrenamiento con ID {task_id} no encontrada.")

        training_task.status = "processing"
        db.commit()
        db.refresh(training_task)

        print(
            f"[{datetime.now()}] Iniciando entrenamiento para la tarea {task_id} (Usuario: {user_id})"
        )
        print(f"[{datetime.now()}] Directorio base de la tarea: {user_task_base_dir}")
        print(f"[{datetime.now()}] Imágenes a procesar: {saved_filenames}")

        instance_count = training_params.get("instance_count", 10)
        class_name = training_params.get("class_name", "my_concept")
        user_prompt = training_params.get("prompt", "default prompt")
        output_name_from_user = training_params.get(
            "output_name", f"model_{model_type}_{task_id}"
        )

        kohya_dataset_dir = os.path.join(
            user_task_base_dir, f"{instance_count}_{class_name}"
        )
        os.makedirs(kohya_dataset_dir, exist_ok=True)
        print(f"[{datetime.now()}] Creando subcarpeta de Kohya_SS: {kohya_dataset_dir}")

        print(
            f"\n--- Movimiento de imágenes y generación de captions con el prompt: '{user_prompt}' ---"
        )
        actual_filenames_in_kohya_dir = []
        for filename in saved_filenames:
            # Los archivos ya están en user_task_base_dir porque fueron extraídos ahí
            original_file_path = os.path.join(user_task_base_dir, filename)
            kohya_target_file_path = os.path.join(kohya_dataset_dir, filename)

            if os.path.exists(original_file_path):
                shutil.move(original_file_path, kohya_target_file_path)
                print(
                    f"   Movido: {filename} de {user_task_base_dir} a {kohya_dataset_dir}"
                )

                base_name = os.path.splitext(filename)[0]
                txt_filename = base_name + ".txt"
                txt_filepath = os.path.join(kohya_dataset_dir, txt_filename)
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(user_prompt)
                print(f"   Creado caption para: {filename} en {kohya_dataset_dir}")
                actual_filenames_in_kohya_dir.append(filename)
            else:
                print(
                    f"   Advertencia: Archivo {original_file_path} no encontrado para mover. Pudo haber sido movido ya o hubo un error en la extracción."
                )

        if not actual_filenames_in_kohya_dir:
            raise Exception(
                "No image files were successfully moved to the Kohya dataset directory."
            )

        print("--- Movimiento y captions completados. ---")

        actual_seed = training_params.get("seed", -1)
        if actual_seed == -1 or actual_seed is None:  # Considerar None también
            actual_seed = random.randint(0, 2**32 - 1)
            print(f"[{datetime.now()}] Generando semilla aleatoria: {actual_seed}")
        else:
            print(f"[{datetime.now()}] Usando semilla proporcionada: {actual_seed}")

        output_dir_for_kohya_models = os.path.join(
            TRAINED_MODELS_DIR, f"user_{user_id}", f"task_{task_id}"
        )
        os.makedirs(output_dir_for_kohya_models, exist_ok=True)

        # La ruta completa al archivo del modelo final
        output_model_path_full = os.path.join(
            output_dir_for_kohya_models, f"{output_name_from_user}.safetensors"
        )

        kohya_script_wrapper_path = os.path.join(KOHYA_SCRIPTS_DIR, "train_lora.py")

        command = [
            sys.executable,  # Python del venv de Celery/API
            kohya_script_wrapper_path,
            "--pretrained_model_name_or_path",
            training_params.get(
                "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
            ),
            "--train_data_dir",
            user_task_base_dir,  # Directorio padre que contiene la carpeta N_classname
            "--output_dir",
            output_dir_for_kohya_models,  # Directorio donde Kohya guarda sus checkpoints y el modelo final
            "--output_name",
            output_name_from_user,
            "--resolution",
            str(training_params.get("resolution", 512)),
            "--train_batch_size",
            str(training_params.get("train_batch_size", 1)),
            "--num_epochs",
            str(training_params.get("num_epochs", 10)),
            "--learning_rate",
            str(training_params.get("learning_rate", 1e-5)),
            "--mixed_precision",
            training_params.get("mixed_precision", "fp16"),
            "--seed",
            str(actual_seed),
            "--network_dim",
            str(training_params.get("network_dim", 128)),
            "--network_alpha",
            str(training_params.get("network_alpha", 64)),
            "--optimizer_type",
            training_params.get("optimizer_type", "AdamW8bit"),
            "--caption_extension",
            ".txt",
            "--lr_scheduler",
            training_params.get("lr_scheduler", "cosine"),
        ]

        if training_params.get("use_xformers", True):
            command.append("--xformers")
        if training_params.get(
            "enable_bucket", True
        ):  # Habilitar bucketing por defecto es bueno
            command.append("--enable_bucket")
        if training_params.get(
            "cache_latents", True
        ):  # Habilitar cache_latents por defecto
            command.append("--cache_latents")
        if training_params.get(
            "bucket_no_upscale", True
        ):  # Habilitar no_upscale por defecto
            command.append("--bucket_no_upscale")

        print(f"\n[{datetime.now()}] Ejecutando comando: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=KOHYA_SCRIPTS_DIR,  # Directorio de trabajo para el script wrapper
        )

        print(f"\n[{datetime.now()}] Iniciando streaming del proceso Kohya_SS...")
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="")  # Esto se logueará en la consola del worker Celery
            process.stdout.close()

        return_code = process.wait()

        if return_code != 0:
            # El error ya se imprimió en el log de Celery gracias al bucle anterior
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output="El error detallado está en el log de Celery.",
            )

        print(
            f"\n[{datetime.now()}] Proceso de Kohya_SS finalizado con código {return_code}."
        )

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
                f"El archivo del modelo no fue encontrado o está vacío en {output_model_path_full} después del entrenamiento."
            )

        training_task.updated_at = datetime.utcnow()
        db.add(training_task)
        db.commit()
        db.refresh(training_task)

    except subprocess.CalledProcessError as e:
        error_message = f"El subproceso de Kohya_SS falló para la tarea {task_id} con código {e.returncode}."
        print(f"[{datetime.now()}] {error_message}", file=sys.stderr)
        if training_task:
            training_task.status = "failed"
            training_task.error_message = (
                error_message  # Opcional: guardar mensaje de error en BD
            )
            training_task.updated_at = datetime.utcnow()
            db.add(training_task)
            db.commit()
        raise  # Re-lanza la excepción para que Celery la marque como fallida

    except Exception as e:
        error_message = (
            f"Error inesperado en la tarea de entrenamiento {task_id}: {str(e)}"
        )
        print(f"[{datetime.now()}] {error_message}", file=sys.stderr)
        # Considerar loguear el traceback completo: import traceback; traceback.print_exc()
        if training_task:
            training_task.status = "failed"
            training_task.error_message = error_message  # Opcional
            training_task.updated_at = datetime.utcnow()
            db.add(training_task)
            db.commit()
        raise  # Re-lanza la excepción

    finally:
        if db:
            db.close()
