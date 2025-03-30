
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load model locally (using distilbert-base-uncased-finetuned-sst-2-english as example)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)

@app.route("/")
def home():
    return "<p>✅ AI Content Detector API is running.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No input text provided."}), 400

        # Split into words and score each (mocked)
        words = text.split()
        highlighted = []
        ai_score_total = 0

        for word in words:
            score = classifier(word)[0][1]["score"]
            ai_score_total += score
            highlighted.append({"word": word, "is_ai": score > 0.6})

        avg_score = ai_score_total / len(words)

        result = "Likely AI-generated" if avg_score > 0.5 else "Likely Human-written"

        return jsonify({
            "result": result,
            "ai_score": round(avg_score, 4),
            "tokens": highlighted
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
