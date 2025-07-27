# --- CRITICAL FIX FOR DEPLOYMENT ---
# eventlet.monkey_patch() must be the first line of code to be executed.
# This patches the standard libraries to be compatible with eventlet's async model.
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import os
import json
import requests
import pytesseract
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# App setup
app = Flask(__name__)
CORS(app, supports_credentials=True)
# Use async_mode='eventlet' for compatibility with the Render start command
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Upload folder
UPLOAD_FOLDER = "/tmp/uploads" # Use a temporary directory for uploads on Render
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("FATAL ERROR: GROQ_API_KEY is not set in environment variables.")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Load system prompt
try:
    with open("prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are a helpful assistant."
    print("Warning: prompt.txt not found. Using a default system prompt.")

# In-memory store for chat history (will reset if the server restarts)
chat_memory = {}

@app.route("/")
def health_check():
    # This route acts as a simple health check for the API.
    return jsonify({"status": "Ben AI Backend is running successfully"}), 200

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    ext = filename.lower().split(".")[-1]
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        if ext == "pdf":
            text = extract_text_from_pdf(save_path)
        elif ext in ["jpg", "jpeg", "png"]:
            text = extract_text_from_image(save_path)
        else:
            return jsonify({"success": False, "error": "Unsupported file type"}), 400
        return jsonify({"success": True, "text": text[:5000]})
    finally:
        # Clean up the uploaded file after processing
        if os.path.exists(save_path):
            os.remove(save_path)


def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_image(path):
    return pytesseract.image_to_string(Image.open(path))

@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

def stream_response_task(sid, user_id, messages, user_message):
    """ The actual task that streams the response. """
    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": messages,
                "stream": True
            },
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line and line.startswith(b"data:"):
                try:
                    data_str = line.decode("utf-8")[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    parsed = json.loads(data_str)
                    content = parsed["choices"][0]["delta"].get("content", "")
                    if content:
                        full_response += content
                        socketio.emit("reply", {"token": content}, room=sid)
                except json.JSONDecodeError:
                    continue
        
        # Update server-side memory with the full conversation turn
        if user_id in chat_memory:
            chat_memory[user_id].append({"role": "user", "content": user_message})
            chat_memory[user_id].append({"role": "assistant", "content": full_response})
        
        socketio.emit("end", {}, room=sid)

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        socketio.emit("error", {"message": f"Could not connect to Groq API: {e}"}, room=sid)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        socketio.emit("error", {"message": str(e)}, room=sid)

@socketio.on("chat")
def handle_chat(data):
    user_id = data.get("user_id", "default")
    user_message = data.get("message", "")
    history = data.get("memory", [])
    sid = request.sid

    if user_id not in chat_memory:
        chat_memory[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    messages_to_send = [chat_memory[user_id][0]] + history
    messages_to_send.append({"role": "user", "content": user_message})

    # Use socketio.start_background_task to run the streaming function
    socketio.start_background_task(
        stream_response_task, 
        sid=sid, 
        user_id=user_id, 
        messages=messages_to_send,
        user_message=user_message
    )

if __name__ == "__main__":
    # This block is for local development only
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
