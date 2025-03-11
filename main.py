
from flask import Flask, request, jsonify, render_template, send_file
import faiss
import json
import os
import numpy as np
import requests
import re
import threading
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
REQUIRED_FILES = ["vector_database.index", "structured_data.json"]
RELEVANCE_THRESHOLD = 1.5
TOP_K_RESULTS = 3
AI_API_KEY = "akm5535x-m60d44cc-39dbca60-fccb8d59"
AI_ENDPOINT_URL = "https://api.us.inc/hanooman/router/v1/chat/completions"
USER_QUERY_HISTORY = {}
QUERY_LOCK = threading.Lock()
DATA_DIR = "C:\\Users\\rishu\\OneDrive\\Desktop\\VizzhyAiAgent\\data"

# Verify required files exist
for file in REQUIRED_FILES:
    if not os.path.exists(file):
        logging.error(f"Required file missing: {file}")
        raise FileNotFoundError(f"Required file missing: {file}")

# Load FAISS index
try:
    index = faiss.read_index("vector_database.index")
    logging.info("FAISS index loaded successfully!")
except Exception as e:
    logging.error(f"Error loading FAISS index: {str(e)}")
    raise e

# Load structured data
try:
    with open("structured_data.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    logging.info(f"Loaded {len(structured_data)} structured text chunks!")
except Exception as e:
    logging.error(f"Error loading structured data: {str(e)}")
    raise e

# Load SentenceTransformer model
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logging.info("Embedding model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading embedding model: {str(e)}")
    raise e


def search_faiss(query, top_k=TOP_K_RESULTS):
    """Search FAISS index for relevant text chunks."""
    try:
        query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = index.search(query_vector, top_k)
        
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(structured_data)]
        retrieved_chunks = [structured_data[idx] for idx in valid_indices]
        
        if not retrieved_chunks or distances[0][0] > RELEVANCE_THRESHOLD:
            return "Out of context", []
        
        return "Relevant", retrieved_chunks
    except Exception as e:
        logging.error(f"FAISS search error: {str(e)}")
        return "Error", []


def query_ai(query, retrieved_chunks, user_id, expected_items=20):
    """Query AI model with retrieved context."""
    context = "\n".join(chunk.get("text_chunk", "") for chunk in retrieved_chunks)
    user_history = "\n".join(USER_QUERY_HISTORY.get(user_id, []))
    
    payload = {
        "model": "everest",
        "messages": [
            {"role": "system", "content": f"You are a chatbot providing responses based ONLY on the given context. Ensure your response contains exactly {expected_items} complete items if requested."},
            {"role": "user", "content": f"User History:\n{user_history}\n\nQuery: {query}\n\nGIVEN CONTEXT:\n{context}"}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(AI_ENDPOINT_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: No valid response.")
        
        # Ensure the response contains expected number of items
        if "genes" in query.lower() and len(re.findall(r"\d+\. ", ai_response)) < expected_items:
            logging.warning("AI response is incomplete. Retrying with strict constraints.")
            return query_ai(query, retrieved_chunks, user_id, expected_items)
        
        return ai_response
    except requests.exceptions.RequestException as e:
        logging.error(f"AI service error: {str(e)}")
        return "Error: AI service is unavailable."


@app.route("/")
def home():
    return render_template("Vizzhy.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default_user")
    
    if not user_message:
        return jsonify({"reply": "Please provide a question or message."})
    
    with QUERY_LOCK:
        USER_QUERY_HISTORY.setdefault(user_id, []).append(user_message)
        if len(USER_QUERY_HISTORY[user_id]) > 10:
            USER_QUERY_HISTORY[user_id] = USER_QUERY_HISTORY[user_id][-10:]
    
    status, retrieved_chunks = search_faiss(user_message)
    if status == "Out of context":
        return jsonify({"reply": "I cannot answer that based on the available data."})
    
    ai_response = query_ai(user_message, retrieved_chunks, user_id, expected_items=20)
    return jsonify({"reply": ai_response})


@app.route("/download/<path:filename>")
def download_file(filename):
    file_path = os.path.join(DATA_DIR, filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) else jsonify({"error": "File not found!"}), 404


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logging.error(f"Flask app failed to start: {str(e)}")

