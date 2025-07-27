from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_session import Session
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import os
import json
import requests
import pytesseract
from PIL import Image
import tempfile

# App setup
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "nova_secret"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
executor = ThreadPoolExecutor()

# Upload folder
UPLOAD_FOLDER = "./pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq API setup
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Load system prompt
with open("prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Memory per user session
chat_memory = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    filename = secure_filename(file.filename)
    ext = filename.lower().split(".")[-1]
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    if ext == "pdf":
        text = extract_text_from_pdf(save_path)
    elif ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(save_path)
    else:
        return jsonify({"success": False, "error": "Unsupported file type"})

    return jsonify({"success": True, "text": text[:5000]})

def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_image(path):
    return pytesseract.image_to_string(Image.open(path))

@socketio.on("chat")
def handle_chat(data):
    user_id = data.get("user_id", "default")
    user_message = data.get("message", "")
    sid = request.sid

    if user_id not in chat_memory:
        chat_memory[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_memory[user_id].append({"role": "user", "content": user_message})

    def stream_response(sid, user_id):
        try:
            response = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": chat_memory[user_id],
                    "stream": True
                },
                stream=True,
                timeout=60
            )

            partial = ""
            for line in response.iter_lines():
                if line and b"data:" in line:
                    try:
                        data = line.decode("utf-8").split("data: ")[-1]
                        parsed = json.loads(data)
                        content = parsed["choices"][0]["delta"].get("content", "")
                        if content:
                            partial += content
                            socketio.emit("reply", {"token": content}, room=sid)
                    except Exception:
                        continue

            chat_memory[user_id].append({"role": "assistant", "content": partial})
            socketio.emit("end", {}, room=sid)

        except Exception as e:
            socketio.emit("error", {"message": str(e)}, room=sid)

    socketio.start_background_task(stream_response, sid, user_id)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
