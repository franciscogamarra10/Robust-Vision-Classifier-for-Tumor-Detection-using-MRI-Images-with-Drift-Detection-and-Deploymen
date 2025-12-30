FROM python:3.10-slim

# 1. Dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalación de PyTorch CPU (Separado para evitar conflictos de index)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Copiamos requerimientos
COPY src/app/requirements.txt ./app_req.txt
COPY src/ml_service/requirements.txt ./ml_req.txt

# 4. Instalamos cada archivo por separado (Esto evita el exit code 2 por encadenamiento)
RUN pip install --no-cache-dir -r app_req.txt
RUN pip install --no-cache-dir -r ml_req.txt
RUN pip install --no-cache-dir gunicorn redis

# 5. Resto del setup
COPY . .
RUN mkdir -p src/static/uploads && chmod -R 777 src/static/uploads

EXPOSE 7860

CMD redis-server --daemonize yes && \
    PYTHONPATH=. python src/ml_service/ml_service.py & \
    gunicorn -b 0.0.0.0:7860 --timeout 120 src.app.app:app
