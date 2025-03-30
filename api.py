from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route("/", methods=["GET"])
def home():
    return "✅ AI Content Detector API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "Text is empty"}), 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
        result = "AI-Generated" if probs[1] > 0.5 else "Human-Written"

        return jsonify({
            "result": result,
            "ai_score": float(probs[1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
