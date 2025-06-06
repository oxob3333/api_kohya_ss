import argparse
import subprocess
import os
import sys
from datetime import datetime

# --- CONFIGURA LA RUTA BASE A TU INSTALACIÓN DE KOHYA_SS ---
KOHYA_SS_PATH = os.path.join(os.path.expanduser("~"), "kohya_ss")

# --- RUTA ABSOLUTA AL INTÉRPRETE DE PYTHON EN TU VENV DE KOHYA_SS ---
KOHYA_SS_PYTHON_EXECUTABLE = os.path.join(KOHYA_SS_PATH, "venv", "bin", "python")

# Asegúrate de que el script principal de entrenamiento de Kohya_SS exista
TRAIN_NETWORK_SCRIPT = os.path.join(KOHYA_SS_PATH, "sd-scripts", "train_network.py")

if not os.path.exists(TRAIN_NETWORK_SCRIPT):
    print(
        f"[{datetime.now()}] ERROR: No se encontró el script de entrenamiento de Kohya_SS en: {TRAIN_NETWORK_SCRIPT}",
        file=sys.stderr,
    )
    sys.exit(1)

if not os.path.exists(KOHYA_SS_PYTHON_EXECUTABLE):
    print(
        f"[{datetime.now()}] ERROR: No se encontró el ejecutable de Python de Kohya_SS Venv en: {KOHYA_SS_PYTHON_EXECUTABLE}",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Invoca el script real de entrenamiento LoRA de Kohya_SS."
    )
    # Argumentos existentes
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Ruta al modelo Stable Diffusion base para entrenar LoRA.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directorio con las imágenes de entrenamiento y sus captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio para guardar el modelo LoRA entrenado.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Nombre del archivo de salida del modelo LoRA (sin extensión).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolución de entrenamiento (ej., 512, 768).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Tamaño del batch de entrenamiento.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento (se mapea a --max_train_epochs en Kohya).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Tasa de aprendizaje para el optimizador.",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=128,
        help="Dimensión de la red LoRA (rank/dim).",
    )
    parser.add_argument(
        "--network_alpha", type=int, default=64, help="Alpha de la red LoRA."
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="AdamW8bit",
        help="Tipo de optimizador (ej., AdamW8bit, Lion, DAdaptAdam).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Precisión mixta (no, fp16, bf16).",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Habilitar el uso de xformers (requiere instalación).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Semilla para la reproducibilidad (-1 para aleatorio).",
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".txt",
        help="Extensión de los archivos de caption.",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="Habilitar el sistema de cubos de Kohya_SS para redimensionamiento inteligente.",
    )

    # --- NUEVOS ARGUMENTOS AÑADIDOS ---
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler (ej. 'cosine', 'constant', 'cosine_with_restarts').",
    )
    parser.add_argument(
        "--cache_latents", action="store_true", help="Habilitar el cacheo de latentes."
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="Deshabilitar el upscaling en los buckets, útil si las imágenes ya tienen el tamaño deseado o se quieren buckets más pequeños.",
    )
    # --- FIN DE NUEVOS ARGUMENTOS ---

    args = parser.parse_args()

    kohya_command = [
        KOHYA_SS_PYTHON_EXECUTABLE,
        TRAIN_NETWORK_SCRIPT,
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--train_data_dir",
        args.train_data_dir,
        "--output_dir",
        args.output_dir,
        "--output_name",
        args.output_name,
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--max_train_epochs",
        str(args.num_epochs),  # Kohya usa max_train_epochs
        "--learning_rate",
        str(args.learning_rate),
        "--network_dim",
        str(args.network_dim),
        "--network_alpha",
        str(args.network_alpha),
        "--optimizer_type",
        args.optimizer_type,
        "--caption_extension",
        args.caption_extension,
        "--save_every_n_epochs",
        "1",
        "--save_model_as",
        "safetensors",
        "--seed",
        str(args.seed),
        "--mixed_precision",
        args.mixed_precision,
        "--network_module",
        "networks.lora",
        # --- AÑADIR NUEVOS ARGUMENTOS AL COMANDO REAL DE KOHYA ---
        "--lr_scheduler",
        args.lr_scheduler,
    ]

    if args.enable_bucket:
        kohya_command.append("--enable_bucket")
    if args.xformers:
        kohya_command.append("--xformers")
    if args.cache_latents:
        kohya_command.append("--cache_latents")
    if args.bucket_no_upscale:
        kohya_command.append("--bucket_no_upscale")
    # --- FIN DE AÑADIR NUEVOS ARGUMENTOS ---

    print(f"\n[{datetime.now()}] Ejecutando comando real de Kohya_SS:")
    print(f"{' '.join(kohya_command)}\n")

    try:
        process = subprocess.run(
            kohya_command, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr
        )
        print(
            f"\n[{datetime.now()}] Entrenamiento de Kohya_SS completado exitosamente."
        )
    except subprocess.CalledProcessError as e:
        print(
            f"\n[{datetime.now()}] --- ¡ERROR! El entrenamiento de Kohya_SS falló. ---",
            file=sys.stderr,
        )
        print(f"Código de salida: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(
            f"\n[{datetime.now()}] --- ERROR INESPERADO al ejecutar Kohya_SS: {e} ---",
            file=sys.stderr,
        )
        sys.exit(1)
