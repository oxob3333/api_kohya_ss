import os
import shutil
import json
import subprocess
from celery import Celery
from sqlalchemy.orm import Session
from datetime import datetime
import sys
import random
import re  # Importar el módulo de expresiones regulares
from typing import List

# Importa desde tus módulos locales
from .database import SessionLocal
from .models import DBTrainingTask

# --- Configuración de Celery ---
# Asegúrate de que el nombre de la app Celery aquí coincida con el usado en app.py para AsyncResult
celery_app = Celery(
    "my_kohya_api_tasks",  # Este nombre es el que usa app.py al importar celery_app
    broker="amqp://guest:guest@localhost:5672//",
    backend="db+sqlite:///./celery_results.db",  # Asegúrate que esta ruta sea accesible por el worker
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,  # Considerar para tareas largas, para que no se re-ejecuten si el worker muere
    task_reject_on_worker_lost=True,  # Con acks_late, si el worker muere, la tarea se re-encola
)

# --- Rutas de Archivos (Ajusta según tu estructura de proyecto real) ---
# Asumiendo que tasks.py está en my_kohya_api/ y la raíz del proyecto contiene la carpeta api_kohya_ss
# Si api_kohya_ss ES la raíz del proyecto (donde está el .git), entonces PROJECT_ROOT sería:
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Y las carpetas uploaded_images y trained_models estarían en api_kohya_ss/uploaded_images etc.

# Asumamos que la estructura es:
# raiz_proyecto/
#   api_kohya_ss/  <-- Aquí está manage.py o el punto de entrada de uvicorn
#     my_kohya_api/  <-- Tu módulo FastAPI
#       tasks.py
#       app.py
#       models.py
#       kohya_scripts/
#         train_lora.py
#     uploaded_images/
#     trained_models/
#     celery_results.db
#     sql_app.db

# Si esta es la estructura, PROJECT_ROOT apunta a la carpeta api_kohya_ss
# --- Rutas de Archivos (Versión CORREGIDA) ---
# Directorio del módulo actual donde está tasks.py (ej. /home/dorian/api_kohya_ss/my_kohya_api)
CURRENT_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raíz del proyecto (un nivel arriba, ej. /home/dorian/api_kohya_ss)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_MODULE_DIR, ".."))

# Rutas de datos (dentro de la carpeta raíz del proyecto)
UPLOADED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "uploaded_images")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

# Ruta a los scripts de Kohya (dentro de la carpeta del módulo my_kohya_api)
KOHYA_SCRIPTS_DIR = os.path.join(CURRENT_MODULE_DIR, "kohya_scripts")


@celery_app.task(
    bind=True, name="my_kohya_api_tasks.train_kohya_model"
)  # Es buena práctica nombrar las tareas
def train_kohya_model(
    self,
    task_id: int,
    user_id: int,
    user_task_base_dir: str,
    model_type: str,
    training_params: dict,
    saved_filenames: List[str],
):
    db = SessionLocal()
    db_task_obj = None

    try:
        db_task_obj = (
            db.query(DBTrainingTask).filter(DBTrainingTask.id == task_id).first()
        )
        if not db_task_obj:
            msg = f"DBTrainingTask {task_id} no encontrada para Celery task {self.request.id}"
            print(f"Error crítico: {msg}")
            self.update_state(
                state="FAILURE",
                meta={
                    "exc_type": "DBTaskNotFound",
                    "exc_message": msg,
                    "progress_text": msg,
                },
            )
            # No se puede actualizar el estado de db_task_obj si no se encuentra
            return  # Termina la tarea Celery

        # Sincronizar el ID de la tarea Celery en la BD si no se guardó antes o cambió
        if db_task_obj.celery_task_id != self.request.id:
            db_task_obj.celery_task_id = self.request.id

        db_task_obj.status = "processing"
        db.commit()  # Guardar celery_task_id y status="processing"

        self.update_state(
            state="PROGRESS",
            meta={
                "progress_text": "Iniciando configuración de Kohya_SS...",
                "progress_percent": 0.0,
            },
        )

        print(
            f"[{datetime.now()}] Iniciando entrenamiento para la tarea de BD {task_id} (Celery ID: {self.request.id}, Usuario: {user_id})"
        )
        print(f"[{datetime.now()}] Directorio base de la tarea: {user_task_base_dir}")

        instance_count = training_params.get("instance_count", 10)
        class_name = training_params.get("class_name", "concept")
        user_prompt = training_params.get("prompt", "a photo of sks style")
        output_name_from_user = training_params.get(
            "output_name", f"model_{model_type}_{task_id}"
        )

        kohya_dataset_dir = os.path.join(
            user_task_base_dir, f"{instance_count}_{class_name}"
        )
        os.makedirs(kohya_dataset_dir, exist_ok=True)
        print(f"[{datetime.now()}] Creando subcarpeta de Kohya_SS: {kohya_dataset_dir}")

        actual_filenames_in_kohya_dir = []
        for filename in saved_filenames:
            original_file_path = os.path.join(user_task_base_dir, filename)
            kohya_target_file_path = os.path.join(kohya_dataset_dir, filename)
            if os.path.exists(original_file_path):
                shutil.move(original_file_path, kohya_target_file_path)
                base_name = os.path.splitext(filename)[0]
                txt_filename = base_name + ".txt"
                txt_filepath = os.path.join(kohya_dataset_dir, txt_filename)
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(user_prompt)
                actual_filenames_in_kohya_dir.append(filename)
            else:
                print(
                    f"   Advertencia: Archivo {original_file_path} no encontrado para mover."
                )

        if not actual_filenames_in_kohya_dir:
            raise Exception(
                "No image files were successfully moved to Kohya dataset directory."
            )

        print(
            f"--- Movimiento y captions para {len(actual_filenames_in_kohya_dir)} imágenes completados. ---"
        )

        actual_seed = training_params.get("seed", -1)
        if actual_seed == -1 or actual_seed is None:
            actual_seed = random.randint(0, 2**32 - 1)

        output_dir_for_kohya_models = os.path.join(
            TRAINED_MODELS_DIR, f"user_{user_id}", f"task_{task_id}"
        )
        os.makedirs(output_dir_for_kohya_models, exist_ok=True)
        output_model_path_full = os.path.join(
            output_dir_for_kohya_models, f"{output_name_from_user}.safetensors"
        )

        kohya_script_wrapper_path = os.path.join(KOHYA_SCRIPTS_DIR, "train_lora.py")

        command = [
            sys.executable,
            kohya_script_wrapper_path,
            "--pretrained_model_name_or_path",
            training_params.get(
                "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
            ),
            "--train_data_dir",
            user_task_base_dir,
            "--output_dir",
            output_dir_for_kohya_models,
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
        if training_params.get("enable_bucket", True):
            command.append("--enable_bucket")
        if training_params.get("cache_latents", True):
            command.append("--cache_latents")
        if training_params.get("bucket_no_upscale", True):
            command.append("--bucket_no_upscale")

        print(
            f"\n[{datetime.now()}] Ejecutando comando para tarea {task_id}: {' '.join(command)}"
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=KOHYA_SCRIPTS_DIR,
        )

        print(
            f"\n[{datetime.now()}] Iniciando streaming del proceso Kohya_SS para tarea {task_id}..."
        )
        last_meaningful_log = "Iniciando entrenamiento..."
        log_buffer = []

        if process.stdout:
            for line_num, line in enumerate(iter(process.stdout.readline, "")):
                print(line, end="")
                cleaned_line = line.strip()
                log_buffer.append(cleaned_line)
                if (
                    len(log_buffer) > 20
                ):  # Mantener solo las últimas 20 líneas, por ejemplo
                    log_buffer.pop(0)

                meta_update = {"progress_text": cleaned_line, "progress_percent": None}
                current_percent = None

                progress_match = re.search(
                    r"steps:\s*(\d+)%\s*\|.*?\|\s*(\d+)/(\d+).*?avr_loss=([\d.]+)",
                    cleaned_line,
                )
                simple_percent_match = re.search(r"^\s*(\d+)\s*%\|", cleaned_line)

                if progress_match:
                    current_percent = float(progress_match.group(1))
                    steps_done = progress_match.group(2)
                    total_steps = progress_match.group(3)
                    avr_loss = progress_match.group(4)
                    meta_update["progress_percent"] = current_percent
                    meta_update["progress_text"] = (
                        f"Paso: {steps_done}/{total_steps} ({current_percent:.1f}%), Pérdida: {avr_loss}"
                    )
                    last_meaningful_log = meta_update["progress_text"]
                elif simple_percent_match:
                    current_percent = float(simple_percent_match.group(1))
                    meta_update["progress_percent"] = current_percent
                    last_meaningful_log = cleaned_line
                elif (
                    "epoch " in cleaned_line.lower()
                    or "saving checkpoint" in cleaned_line.lower()
                    or "loading model" in cleaned_line.lower()
                    or "caching latents" in cleaned_line.lower()
                    or "model saved." in cleaned_line
                ):
                    last_meaningful_log = cleaned_line
                    meta_update["progress_text"] = last_meaningful_log

                # Solo actualizar el estado si hay un cambio significativo o cada N líneas para no sobrecargar
                if current_percent is not None or (
                    line_num % 10 == 0
                    and meta_update["progress_text"] != last_meaningful_log
                ):
                    self.update_state(state="PROGRESS", meta=meta_update)
            process.stdout.close()

        return_code = process.wait()

        if return_code != 0:
            full_log_on_failure = "\n".join(log_buffer)
            failure_reason = f"Subproceso Kohya falló con código {return_code}. Últimos logs: ...{full_log_on_failure[-1000:]}"  # Últimos 1000 caracteres
            self.update_state(
                state="FAILURE",
                meta={
                    "exc_type": "SubprocessError",
                    "exc_message": failure_reason,
                    "progress_text": failure_reason,
                },
            )
            db_task_obj.status = "failed"
            db_task_obj.error_message = failure_reason
            db.commit()
            raise subprocess.CalledProcessError(
                return_code, command, output=failure_reason
            )

        print(
            f"\n[{datetime.now()}] Proceso Kohya_SS para tarea {task_id} finalizado con código {return_code}."
        )

        if (
            os.path.exists(output_model_path_full)
            and os.path.getsize(output_model_path_full) > 0
        ):
            db_task_obj.status = "completed"
            db_task_obj.output_model_path = output_model_path_full
            db_task_obj.error_message = None
            final_message = (
                f"Tarea {task_id} completada. Modelo: {output_model_path_full}"
            )
            print(f"[{datetime.now()}] {final_message}")
            self.update_state(
                state="SUCCESS",
                meta={
                    "progress_text": "Completado exitosamente!",
                    "progress_percent": 100.0,
                    "output_model_path": output_model_path_full,
                },
            )
        else:
            final_error = f"Modelo no encontrado/vacío: {output_model_path_full}"
            self.update_state(
                state="FAILURE",
                meta={
                    "exc_type": "ModelNotFound",
                    "exc_message": final_error,
                    "progress_text": final_error,
                },
            )
            db_task_obj.status = "failed"
            db_task_obj.error_message = final_error
            raise Exception(final_error)

        db_task_obj.updated_at = datetime.utcnow()
        db.add(db_task_obj)
        db.commit()

    except subprocess.CalledProcessError as e:
        error_message = f"Subproceso Kohya_SS (CalledProcessError) para tarea {task_id}: {str(e.output or e)}"
        print(f"[{datetime.now()}] {error_message}", file=sys.stderr)
        if db_task_obj:
            db_task_obj.status = "failed"
            db_task_obj.error_message = error_message
            db.commit()
        # El estado de Celery ya debería estar en FAILURE si se lanzó desde el bloque if return_code != 0
        # Si no, lo actualizamos aquí.
        if self.request.state != "FAILURE":
            self.update_state(
                state="FAILURE",
                meta={
                    "exc_type": type(e).__name__,
                    "exc_message": str(e.output or e),
                    "progress_text": "Fallo en subproceso.",
                },
            )
        raise
    except Exception as e:
        error_message = f"Error inesperado en tarea {task_id}: {str(e)}"
        print(f"[{datetime.now()}] {error_message}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        if db_task_obj:
            db_task_obj.status = "failed"
            db_task_obj.error_message = error_message
            db.commit()
        self.update_state(
            state="FAILURE",
            meta={
                "exc_type": type(e).__name__,
                "exc_message": str(e),
                "progress_text": "Error inesperado en la tarea.",
            },
        )
        raise
    finally:
        if db:
            db.close()
