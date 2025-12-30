import os
import json
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import redis
import numpy as np


# Configuración básica
#REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "queue:requests")
UPLOAD_FOLDER = "/app/static/uploads"
#MODEL_PATH = "models/best_model_f2.pt"
MODEL_PATH = "src/ml_service/models/best_model_f2.pt"
#BASELINE_PATH = "models/train_baseline.npy"
BASELINE_PATH = "src/ml_service/models/train_baseline.npy"
# Optimización para Linux: Evita que PyTorch consuma RAM de más
torch.set_num_threads(1)

class FineTuneMobileNetV2(nn.Module):
    def __init__(self, num_classes, input_size, dropout=0.3, use_bn=True, head_units=1024, use_gap=True):
        super().__init__()
        full = models.mobilenet_v2(weights=None)
        self.backbone = full.features
        self.last_channel = full.last_channel
        self.bn = nn.BatchNorm2d(self.last_channel) if use_bn else nn.Identity()
        self.use_gap = use_gap

        self.fc = nn.Sequential(
            nn.Linear(self.last_channel, head_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_units, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        if self.use_gap:
            x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Inicializar modelo
device = torch.device("cpu")
model = FineTuneMobileNetV2(num_classes=2, input_size=224)

baseline_data = np.load(BASELINE_PATH)
    # Calculamos los parámetros estadísticos una sola vez al arrancar
BASE_MEAN = np.mean(baseline_data)
BASE_STD = np.std(baseline_data)
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # Desactivamos gradientes globalmente para ahorrar memoria
    torch.set_grad_enabled(False)
    print(f"✅ Modelo cargado exitosamente desde {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR cargando modelo: {e}")

transform = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Conexión a Redis
db = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def main():
    print("🚀 ML Service iniciado y esperando tareas...")
    while True:
        # Bloquea hasta que haya un mensaje
        item = db.brpop(REDIS_QUEUE, timeout=30)
        if not item:
            continue
        
        msg = json.loads(item[1])
        job_id = msg["id"]
        filename = msg["filename"]
        path = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"📦 Procesando: {filename}")
        
        try:
            # Procesamiento de imagen
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0)
            # --- CÁLCULO DE DRIFT (Z-SCORE) ---
            current_mean = x.mean().item()
            # Cuántas desviaciones estándar se aleja la imagen actual de la media base
            z_score = abs(current_mean - BASE_MEAN) / (BASE_STD )
            
            # Lógica de decisión por Drift
            drift_msg = ""
            if z_score > 3.0:  # Umbral estadístico: Fuera del 99.7% de la distribución normal
                prediction = [f"Error: La imagen no parece ser una resonancia válida (Drift Crítico).{z_score}"]
            else:
                if z_score > 1.5 and z_score <3: # Umbral de advertencia
                    drift_msg = " [⚠️ Warning: Calidad o contraste inusual detectado]"
                
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
                score, idx = torch.max(probs, dim=0)
                label = "Tumor" if idx.item() == 1 else "No tumor"
                prediction = [f"{label} (prob={score.item():.3f}){drift_msg}"]

            db.set(job_id, json.dumps({"prediction": prediction}))
            print(f"✅ {filename} procesado. Z-Score: {z_score:.2f}")
            
            # Predicción pura
            #out = model(x)
            #probs = torch.softmax(out, dim=1)[0]
            #score, idx = torch.max(probs, dim=0)
            
            #label = "Tumor" if idx.item() == 1 else "No tumor"
            #prediction = [f"{label} (prob={score.item():.3f})"]
            
            # Enviar resultado de vuelta a Redis
            #db.set(job_id, json.dumps({"prediction": prediction}))
            print(f"✅ Resultado enviado para {job_id}")

        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            db.set(job_id, json.dumps({"prediction": [f"Error: {str(e)}"]}))

if __name__ == "__main__":
    main()
