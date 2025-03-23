from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load tokenizer and model
MODEL_NAME = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.route("/")
def index():
    return "<p style='color:green;'>✅ AI Content Detector API is running.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        ai_score = probs[0][1].item()  # Class 1 is AI-generated

    result = "AI-Generated" if ai_score > 0.5 else "Human-Written"

    return jsonify({
        "ai_score": ai_score,
        "result": result
    })

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=10000)