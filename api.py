from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# 👇 Add this line to allow only your domain (for security)
CORS(app, origins=["https://theaicontentdetector.com"])

# Load model and tokenizer
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route("/")
def home():
    return "<p>✅ AI Content Detector API is running.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    score = torch.sigmoid(outputs.logits)[0].item()
    result = "AI-Generated" if score > 0.5 else "Human-Written"
    return jsonify({"ai_score": score, "result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
