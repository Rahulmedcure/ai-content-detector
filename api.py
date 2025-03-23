from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os  # ✅ Added to access PORT environment variable

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# Home route to check if API is working
@app.route('/')
def home():
    return "✅ AI Content Detector API is running."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    ai_score = float(scores[0][1])  # class 1 = AI-generated
    return jsonify({
        "ai_score": ai_score,
        "result": "AI-Generated" if ai_score > 0.5 else "Human-Written"
    })

# Run app with correct host and port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
