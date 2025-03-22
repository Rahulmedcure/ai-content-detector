from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# Add this route so the base URL shows something
@app.route('/')
def home():
    return "✅ AI Content Detector API is running."

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

if __name__ == '__main__':
    app.run(debug=True)
