from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re

app = Flask(__name__)
CORS(app)

# Load model + tokenizer
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Token limit check (max 500 tokens safely)
        tokenized = tokenizer.encode(text, truncation=False)
        if len(tokenized) > 500:
            return jsonify({"error": "Text too long. Please submit up to 500 tokens."}), 400

        # Split text into sentences
        sentences = split_into_sentences(text)
        highlights = []
        total_score = 0

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                ai_score = probs[0][1].item()
                total_score += ai_score
                highlights.append({
                    "text": sentence,
                    "is_ai": ai_score > 0.6,
                    "ai_score": round(ai_score, 3)
                })

        avg_score = total_score / len(highlights)
        result = "Likely AI" if avg_score > 0.6 else "Likely Human"

        return jsonify({
            "result": result,
            "ai_score": round(avg_score, 3),
            "highlights": highlights
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong."}), 500
