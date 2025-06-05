# API REST para Entrenamiento Remoto de Modelos LoRA con Kohya_SS

Este proyecto implementa una API REST completa y asíncrona construida con **FastAPI** para gestionar el entrenamiento de modelos de difusión (específicamente LoRAs) de forma remota utilizando los potentes scripts de **Kohya_SS**.

El sistema está diseñado para ser operado en un entorno de servidor, permitiendo a los usuarios enviar trabajos de entrenamiento, monitorear su progreso y descargar los modelos resultantes, todo a través de endpoints seguros.

## ✨ Características Principales

- **Entrenamiento Remoto:** Lanza trabajos de entrenamiento de Kohya_SS a través de una API, sin necesidad de interactuar directamente con la línea de comandos del entrenador.
- **Procesamiento Asíncrono:** Utiliza **Celery** como gestor de tareas y **RabbitMQ** como message broker para manejar los entrenamientos (que son procesos largos) en segundo plano, manteniendo la API siempre rápida y responsiva.
- **Autenticación y Seguridad:** Sistema de usuarios con autenticación basada en tokens JWT (OAuth2).
- **Gestión de Usuarios:** Un superusuario predefinido puede registrar nuevos usuarios, controlando quién tiene acceso al sistema.
- **Subida de Datasets Eficiente:** Los usuarios pueden subir un archivo `.zip` con todas sus imágenes de entrenamiento, que el servidor procesa automáticamente.
- **Monitorización de Tareas en Tiempo Real:** Un endpoint de estado permite consultar el progreso del entrenamiento, mostrando el porcentaje y el último log relevante de Kohya_SS.
- **Optimización para Rendimiento:** Diseñado para correr en un entorno WSL 2 (Subsistema de Windows para Linux) para aprovechar al máximo la GPU con librerías como Triton y xformers.

## ⚙️ Arquitectura del Sistema

El sistema se compone de varios servicios desacoplados que trabajan en conjunto:

1.  **API FastAPI (Uvicorn):** Es el punto de entrada. Recibe las peticiones web de los usuarios, gestiona la autenticación, interactúa con la base de datos y envía las tareas de entrenamiento a RabbitMQ.
2.  **RabbitMQ (Message Broker):** Actúa como un intermediario o "cartero". Recibe los mensajes de tarea de la API y los guarda en una cola hasta que un worker esté disponible.
3.  **Worker(s) de Celery:** Son los trabajadores de fondo. Escuchan constantemente la cola de RabbitMQ. Cuando llega una nueva tarea, la toman y ejecutan el proceso de entrenamiento de Kohya_SS, utilizando la GPU.
4.  **Base de Datos (SQLAlchemy + SQLite):** Almacena la información de los usuarios y el estado de las tareas de entrenamiento.

**Flujo de un Trabajo de Entrenamiento:**
`Usuario -> API FastAPI -> (Mensaje) -> RabbitMQ [Cola] -> (Mensaje) -> Worker Celery -> Lanza Kohya_SS en GPU`

## 🚀 Guía de Instalación y Ejecución

Este sistema está diseñado para correr en un entorno **WSL 2** para un rendimiento óptimo, pero puede ser usado en entornos de linux sin problema.

### Prerrequisitos

- Windows con WSL 2 instalado y una distribución de Linux (ej. Ubuntu).
- NVIDIA GPU con drivers CUDA correctamente instalados y accesibles desde WSL 2.
- Python 3.10 o superior instalado en WSL 2 (incluyendo `python3.10-dev`).
- Git.

### Pasos de Instalación

1.  **Clonar Repositorios:**
    Dentro de tu terminal de WSL, clona este repositorio y el de Kohya_SS:

    ```bash
    # Clona tu API
    git clone [https://github.com/oxob3333/api_kohya_ss.git](https://github.com/oxob3333/api_kohya_ss.git)

    # Clona Kohya_SS
    git clone [https://github.com/bmaltais/kohya_ss.git](https://github.com/bmaltais/kohya_ss.git)
    ```

2.  **Configurar Entorno de Kohya_SS:**
    Crea un entorno virtual dentro de la carpeta de Kohya y activa las dependencias.

    ```bash
    cd kohya_ss
    python3.10 -m venv venv
    source venv/bin/activate

    # Instalar dependencias (usar el script es lo más seguro)
    chmod +x setup.sh
    ./setup.sh

    # Instalar Triton
    pip install triton
    ```

3.  **Configurar la API:**
    Con el mismo entorno virtual activado, instala las dependencias de la API.

    ```bash
    # Vuelve a la carpeta de tu API
    cd ../api_kohya_ss

    # Instala las librerías del requirements.txt
    pip install -r requirements.txt
    ```

4.  **Configurar el Superusuario (Opcional):**
    Puedes definir las credenciales del superusuario y la `SECRET_KEY` de la API como variables de entorno antes de iniciar la aplicación.
    ```bash
    export MASTER_USER_USERNAME="tu_admin"
    export MASTER_USER_PASSWORD="tu_super_password"
    export SECRET_KEY="una_clave_secreta_muy_larga_y_segura"
    ```
    Si no se definen, la aplicación usará los valores por defecto definidos en `app.py` y `auth.py`.

### Ejecutar el Sistema

Necesitarás **dos terminales de WSL** abiertas, ambas con el entorno virtual activado (`source ~/kohya_ss/venv/bin/activate`).

**Terminal 1: Iniciar el Worker de Celery**

```bash
# Navega a la carpeta de la API
cd ~/api_kohya_ss

# Inicia el worker
celery -A my_kohya_api.tasks worker --loglevel=info -P solo
```

# Navega a la carpeta de la API

cd ~/api_kohya_ss

# Inicia el servidor Uvicorn

uvicorn my_kohya_api.app:app --host 0.0.0.0 --port 8000
