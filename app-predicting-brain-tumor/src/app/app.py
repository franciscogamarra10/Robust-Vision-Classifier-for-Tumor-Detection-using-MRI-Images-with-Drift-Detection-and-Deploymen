import os
import time
import json
from uuid import uuid4
from flask import Flask, Blueprint, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import redis

# Configuración
API_SLEEP = float(os.getenv("API_SLEEP", 0.5))
RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", 30))
#REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "queue:requests")

UPLOAD_FOLDER = "/app/static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
FEEDBACK_FILE = os.path.join(UPLOAD_FOLDER, "feedback.txt")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "mri-secret")
app_router = Blueprint("app_router", __name__, template_folder="templates")
db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict_via_redis(filename: str):
    job_id = str(uuid4())
    job_data = {"id": job_id, "filename": filename}
    db.lpush(REDIS_QUEUE, json.dumps(job_data))

    start = time.time()
    while (time.time() - start) < RESPONSE_TIMEOUT:
        result = db.get(job_id)
        if result:
            msg = json.loads(result)
            db.delete(job_id)
            return msg.get("prediction", ["Error en proceso"])
        time.sleep(API_SLEEP)
    return ["Timeout esperando al servicio de ML"]

@app_router.route("/", methods=["GET", "POST"])
def recommend_books():
    if request.method == "GET":
        return render_template("index.html")

    if 'image' not in request.files:
        return render_template("index.html", error="No file part")
    
    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", error="No selected file")

    if not allowed_file(file.filename):
        return render_template("index.html", error="File type not allowed")

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    recommendations = model_predict_via_redis(filename)
    return render_template("index.html", recommendations=recommendations)

@app_router.route("/feedback", methods=["POST"])
def feedback():
    data = request.form.get("feedback")
    if not data:
        return render_template("index.html", error="Please enter feedback")

    feedback_id = str(uuid4())
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(f"{feedback_id}\t{timestamp}\t{data}\n")

    return render_template("index.html", success="Thanks for your feedback!")

app.register_blueprint(app_router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
